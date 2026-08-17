#!/usr/bin/env python3
"""Run the AUTHORS' OWN code (Reddy & Challaram 2026, "Don't Ask the LLM to
Track Freshness") under OUR experimental setup, for a fair comparison.

We import their verbatim pipeline functions from their public repo
(cvikasreddy/memory-conflict-resolution scripts/_pipeline.py):
    bm25_retrieve, _extract_candidates (their exact CANDIDATE_PROMPT),
    _freshness_pick (deterministic max(serial)).
Their code is Langfuse-instrumented; we inject a no-op `_lf` stub so no
Langfuse/network is needed and no .env is read.

OUR setup (experiment.md §4.1), i.e. what we hold to compare fairly:
  - fact bank   = OURS' P1 extraction (extraction-controlled, like baseline `b`)
  - freshness   = OURS' ingestion ORDINAL (running index in ingestion order;
                  validated new_ord>old_ord 62/62 on 6k has_pair) prepended as
                  the "serial" the authors' prompt copies.
  - retrieval   = the authors' BM25 top-10 (their method's own design; disclosed
                  the same way we disclose Zep's k=10 vs ours' k=100).
  - metric      = OUR official MAB SubEM (substring_exact_match via
                  default_post_process: max over raw/parsed), reported OVERALL-100
                  (canonical) + has_pair (mechanism).
  - backbone    = PIPELINE_MODEL (the authors' extraction LLM = our backbone).

Set THEIR_REPO to the local clone (default: sibling of MemoryAgentBench).

Usage:
  # A) OpenAI backbone (default):
  export PIPELINE_MODEL=gpt-4o-mini
  set -a; . .env; set +a; export OPENAI_API_KEY="$OPENAI_API_KEY_A"
  python maxserial_theircode.py --length 6k

  # B) gemma via Ollama (embed stays OpenAI, chat swaps to Ollama):
  export PIPELINE_MODEL=gemma3:4b
  export OLLAMA_CHAT_URL=http://localhost:11434/v1   # triggers dual-client mode
  set -a; . .env; set +a; export OPENAI_API_KEY="$OPENAI_API_KEY_A"  # embed only
  python maxserial_theircode.py --length 6k

Dual-client mode (OLLAMA_CHAT_URL set):
  - embeddings still call OpenAI text-embedding-3-small (bank_emb npy cache shared across backbones)
  - chat.completions.create routes to OLLAMA_CHAT_URL with model=PIPELINE_MODEL
  - filename model tag sanitized (`:` → `-`); output: {L}_{gemma3-Xb}_vector100.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# authors' repo is vendored in-repo (MIT); fall back to a sibling clone / env override
_THEIR_CANDIDATES = [
    REPO / "dont_ask/memory-conflict-resolution",
    REPO.parent / "memory-conflict-resolution",
]
_env = os.environ.get("THEIR_REPO")
THEIR_REPO = (Path(_env) if _env else
              next((p for p in _THEIR_CANDIDATES if p.exists()), _THEIR_CANDIDATES[0]))


# ── inject a no-op `_lf` so we can import the authors' instrumented pipeline ──
def _install_lf_stub():
    from openai import OpenAI  # plain client (no Langfuse wrapper)

    class _NoopClient:
        def update_current_span(self, *a, **k): pass
        def update_current_generation(self, *a, **k): pass
        def score_current_trace(self, *a, **k): pass
        def score_current_span(self, *a, **k): pass

    def observe(*d_args, **d_kwargs):
        def deco(fn):
            return fn
        # support both @observe and @observe(name=...)
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return deco

    stub = types.ModuleType("_lf")
    stub.OpenAI = OpenAI
    stub.observe = observe
    stub.get_client = lambda: _NoopClient()
    stub.ROOT = THEIR_REPO
    stub._lf = None
    sys.modules["_lf"] = stub


_install_lf_stub()
sys.path.insert(0, str(THEIR_REPO / "scripts"))
import _pipeline as A  # authors' verbatim pipeline (_extract_candidates = their CANDIDATE_PROMPT, _freshness_pick = max serial)
import numpy as np
from rank_bm25 import BM25Okapi

EMBED_MODEL = "text-embedding-3-small"  # OUR embedder (experiment.md §4.1.5)


# ── OUR retrieval: raw-question vector search over the existing fact bank ─────
# (fairness: SAME retrieval as ours — raw-q, text-embedding-3-small, top-100.
#  We keep the authors' resolution (extract+max); only the retrieval is aligned
#  to ours, so the sole method difference is identity-resolution.)
def _embed(client, texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        r = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in r.data)
    a = np.asarray(out, dtype=np.float32)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    return a


def _load_or_build_bank_emb(client, length, fact_texts):
    # bank_emb cache is keyed by the bank source: gpt-4o-mini P1 (default, shared
    # across backbones) vs a per-backbone gemma P1 bank (MAXSERIAL_BANK_TAG set) —
    # different fact texts -> different embeddings -> must not collide.
    _tag = os.environ.get("MAXSERIAL_BANK_TAG", "")
    _sfx = f"_{_tag}" if _tag else ""
    p = REPO / f"outputs/maxserial_theircode/bank_emb_{length}{_sfx}.npy"
    if p.exists():
        emb = np.load(p)
        if emb.shape[0] == len(fact_texts):
            return emb
    emb = _embed(client, fact_texts)
    os.makedirs(p.parent, exist_ok=True)
    np.save(p, emb)
    return emb


def vector_retrieve(qemb, bank_emb, fact_indices, fact_texts, top_k):
    sims = bank_emb @ qemb  # cosine (both L2-normalized)
    top = np.argsort(sims)[::-1][:top_k]
    return [{"rank": r + 1, "fact_idx": int(fact_indices[i]), "score": float(sims[i]),
             "text": fact_texts[i]} for r, i in enumerate(top)]


# ── OUR data: bank (P1 extraction, ordinal) + queries ────────────────────────
def load_bank(length):
    # Fact bank = P1 extraction, SHARED across all methods (ours / Don't Ask /
    # Vanilla-RAG / mem0). Default = gpt-4o-mini P1. For per-backbone deployment,
    # MAXSERIAL_BANK_CACHE points at that backbone's own P1 extraction (e.g.
    # analysis/results/p1_caches__gemma3-4b/extraction_cache_p1_6k.json) so the
    # candidate-extraction LLM operates on the SAME facts our pipeline extracted.
    bank_path = os.environ.get("MAXSERIAL_BANK_CACHE") or str(
        REPO / f"analysis/results/p1_caches/extraction_cache_p1_{length}.json")
    cache = json.load(open(bank_path, encoding="utf-8"))
    fact_indices, fact_texts = [], []
    for facts in cache.values():
        for f in facts:
            fact_indices.append(len(fact_texts))  # ordinal = running index
            fact_texts.append(f)
    return fact_indices, fact_texts


def load_queries(length):
    return json.load(open(
        REPO / f"analysis/results/sh_{length}_mquake_analysis.json", encoding="utf-8"))


# ── OUR official metric (MAB SubEM via default_post_process) ──────────────────
def official_subem(prediction, gt_answer):
    from utils.eval_other_utils import (
        substring_exact_match_score, drqa_metric_max_over_ground_truths, parse_output,
    )
    gts = gt_answer if isinstance(gt_answer, list) else [gt_answer]
    gts = [str(g) for g in gts if g is not None]
    def sc(p):
        return bool(drqa_metric_max_over_ground_truths(substring_exact_match_score, p, gts))
    raw = sc(prediction)
    parsed = parse_output(prediction)
    return raw or (sc(parsed) if parsed is not None else False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="6k")
    ap.add_argument("--retrieval", choices=["vector", "bm25"], default="vector",
                    help="vector = OUR raw-q text-embedding-3-small top-100 (default; "
                         "SAME retrieval as ours -> cross-comparable without redoing "
                         "ours at top-10; the shared 100-cand pool isolates resolution); "
                         "bm25 = authors' as-designed top-10 sensitivity")
    ap.add_argument("--topk", type=int, default=0, help="0 = 100 (vector) / 10 (bm25)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    topk = args.topk or (100 if args.retrieval == "vector" else 10)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set. Source .env + export it.")
    model = A.MODEL  # authors read PIPELINE_MODEL at import; use their actual value

    fact_indices, fact_texts = load_bank(args.length)
    queries = load_queries(args.length)
    if args.limit:
        queries = queries[: args.limit]

    # Strip authors' `name="..."` kwarg (Langfuse-wrapped-client convention);
    # the plain OpenAI SDK client does not accept it.
    def _patch_strip_name(cli, inject=None):
        # Strip authors' name= kwarg. `inject` (Ollama-only) adds default kwargs, e.g.
        # max_tokens, to bound weak-backbone JSON-mode runaway generation (gemma-1b
        # extracts a candidate for ALL 100 pool facts -> ~1.8-6.7k tokens; disclosed
        # weak-backbone deviation, applied ONLY to the Ollama chat client).
        _comp = cli.chat.completions
        _orig = _comp.create
        def _create(*a, **kw):
            kw = {k: v for k, v in kw.items() if k != "name"}
            for k, v in (inject or {}).items():
                kw.setdefault(k, v)
            return _orig(*a, **kw)
        _comp.create = _create

    embed_client = A.OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # OpenAI, embeddings
    ollama_chat_url = os.environ.get("OLLAMA_CHAT_URL")
    if ollama_chat_url:
        # Dual-client: chat.completions routed to Ollama (OpenAI-compatible endpoint);
        # embeddings stay on real OpenAI (Ollama has no text-embedding-3-small analogue).
        chat_client = A.OpenAI(api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
                               base_url=ollama_chat_url)
        # UNIFORM EXCEPTION POLICY (weak-backbone; see experiment.md appendix):
        #  - max_tokens bounds runaway generation (a single call must not ramble
        #    over all 100 retrieved facts).
        #  - timeout is a per-call total wall budget: beyond it we DECLARE the
        #    candidate-extraction step failed (resolves "still reasoning vs hung"
        #    by definition; the try/except in the loop maps it to empty extraction).
        _patch_strip_name(chat_client, inject={
            "max_tokens": int(os.environ.get("MAXSERIAL_MAX_TOKENS", "4096")),
            "timeout": float(os.environ.get("MAXSERIAL_TIMEOUT", "300")),
        })
        print(f"[dual-client] embed=OpenAI(text-embedding-3-small) | "
              f"chat={ollama_chat_url} model={model}")
    else:
        chat_client = embed_client
        _patch_strip_name(chat_client)

    if args.retrieval == "vector":
        bank_emb = _load_or_build_bank_emb(embed_client, args.length, fact_texts)
        q_emb = _embed(embed_client, [q["question"] for q in queries])
    else:
        bm25 = BM25Okapi([A.tokenize(t) for t in fact_texts])
    print(f"[authors' method × OUR setup] bank={len(fact_texts)} | queries={len(queries)} "
          f"| model={model} | retrieval={args.retrieval} topk={topk} | THEIR_REPO={THEIR_REPO.name}")

    rows = []
    n = ncorr = 0
    hp = hpcorr = 0
    n_empty = 0
    step_stats = {}   # candidate-extraction step failure attribution (uniform policy)
    for i, q in enumerate(queries):
        # --- retrieval: OUR raw-q vector (fair) or authors' bm25 (cross-check) ---
        if args.retrieval == "vector":
            retrieved = vector_retrieve(q_emb[i], bank_emb, fact_indices, fact_texts, topk)
        else:
            retrieved = A.bm25_retrieve(bm25, q["question"], fact_indices, fact_texts, topk)
        # --- authors' verbatim resolution (their CANDIDATE_PROMPT + max serial) ---
        # UNIFORM POLICY: run the step unchanged, but wrap it so a weak-backbone
        # exception (timeout / API error) becomes a DECLARED step-failure rather
        # than a crash or unbounded hang. Timing lets us post-hoc separate
        # runaway (slow, tokens flowing) from hang (timeout with no output).
        _t0 = time.time()
        _fail = None
        try:
            cands_raw = A._extract_candidates(chat_client, q["question"], retrieved)
        except Exception as e:                       # noqa: BLE001 (declare step-fail)
            cands_raw = []
            _fail = type(e).__name__                 # e.g. APITimeoutError / APIError
        _elapsed = round(time.time() - _t0, 2)
        # Malformed-schema guard (disclosed): _parse_candidates does NOT validate
        # per-candidate keys; weak backbones emit candidates missing "serial"/
        # "answer_entity" or a non-int serial, which would crash _freshness_pick.
        # Drop malformed candidates; none-left -> None -> "no answer" (empty).
        cands = []
        for _c in cands_raw:
            if not (isinstance(_c, dict) and "answer_entity" in _c and "serial" in _c):
                continue
            try:
                _c = {**_c, "serial": int(_c["serial"])}
            except (TypeError, ValueError):
                continue
            cands.append(_c)
        n_dropped = len(cands_raw) - len(cands)
        # Attribute this query's candidate-extraction step (uniform categories):
        if _fail:
            cx_status = _fail                        # timeout / api-error
        elif len(cands_raw) == 0:
            cx_status = "empty_parse"                # returned but 0 parseable
        elif len(cands) == 0:
            cx_status = "malformed_all"              # all candidates schema-invalid
        elif n_dropped:
            cx_status = "partial_malformed"          # some usable, some dropped
        else:
            cx_status = "ok"
        step_stats[cx_status] = step_stats.get(cx_status, 0) + 1
        chosen = A._freshness_pick(cands)
        answer = (chosen["answer_entity"] if chosen else "no answer") or "no answer"
        # --- our official metric ---
        if answer == "no answer":
            n_empty += 1
            correct = False
        else:
            correct = official_subem(answer, q.get("gt_answer"))
        n += 1; ncorr += correct
        if q.get("conflict_type") == "has_pair":
            hp += 1; hpcorr += correct
        rows.append({
            "query_id": q.get("query_id"), "conflict_type": q.get("conflict_type"),
            "retrieved_ords": [r["fact_idx"] for r in retrieved],
            "n_candidates": len(cands),
            "chosen_serial": chosen["serial"] if chosen else None,
            "answer": answer, "gt_answer": q.get("gt_answer"), "subem": correct,
            "cand_extract_status": cx_status, "cand_extract_sec": _elapsed,
            "n_malformed_dropped": n_dropped,
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(queries)}")

    print(f"\n=== authors' method × OUR setup  {args.length} × {model} "
          f"(retrieval={args.retrieval} top{topk}, official SubEM) ===")
    print(f"overall : {ncorr}/{n} ({100*ncorr/n:.1f}%)   <- canonical (experiment.md §4.1.4)")
    print(f"has_pair: {hpcorr}/{hp} ({100*hpcorr/hp:.1f}%)")
    print(f"empty extraction: {n_empty}/{n} ({100*n_empty/n:.1f}%)")
    # candidate-extraction step attribution (uniform exception policy)
    _secs = [r["cand_extract_sec"] for r in rows]
    print("cand-extract step: " + " | ".join(f"{k}={v}" for k, v in sorted(step_stats.items()))
          + f"  (sec: max={max(_secs):.1f} mean={sum(_secs)/len(_secs):.1f}"
          + f" | timeout={os.environ.get('MAXSERIAL_TIMEOUT','300')}s"
          + f" max_tokens={os.environ.get('MAXSERIAL_MAX_TOKENS','4096')})")

    # Sanitize model tag for filename cross-platform (gemma3:4b -> gemma3-4b)
    model_tag = model.replace(":", "-").replace("/", "-")
    out = args.out or str(REPO / f"outputs/maxserial_theircode/"
                          f"{args.length}_{model_tag}_{args.retrieval}{topk}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"length": args.length, "model": model, "source": "authors_pipeline",
               "retrieval": args.retrieval, "topk": topk,
               "overall": [ncorr, n], "has_pair": [hpcorr, hp], "n_empty": n_empty,
               "bank_cache": os.environ.get("MAXSERIAL_BANK_CACHE", "gpt-4o-mini(default)"),
               "policy": {"timeout_s": float(os.environ.get("MAXSERIAL_TIMEOUT", "300")),
                          "max_tokens": int(os.environ.get("MAXSERIAL_MAX_TOKENS", "4096"))},
               "cand_extract_step_stats": step_stats,
               "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
