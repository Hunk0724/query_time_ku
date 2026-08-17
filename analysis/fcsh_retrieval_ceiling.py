#!/usr/bin/env python3
"""FC-SH retrieval ceiling vs end-to-end EM (has_pair, gpt-4o-mini, 4 lengths).

PURPOSE (paper: rule out "a stronger retriever would solve FC-SH").
    All methods share ONE retrieval (Mem0 bank + vector top-100). So retrieval
    quality is a SINGLE shared ceiling, measured on the FAITHFUL-write pool
    (struct dir = all versions in bank) so it isolates RETRIEVAL (does vector
    top-100 surface gt_new when it exists) from write-time deletion.

    Ceiling  = % has_pair where gt_new is in the top-100 pool (fact-text match).
    Achieved = each method's has_pair sEM.
    If ceiling ~100% but a method sits far below, its gap is DOWNSTREAM KU
    judgment, not retrieval -> a stronger retriever has ~0 headroom.

Committed data only; no model re-run; no .env. Reuses canonical primitives.
    python analysis/fcsh_retrieval_ceiling.py
"""
import glob, json, os, sys
from pathlib import Path
from collections import Counter
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))
from analysis.rescore_canonical import load_haspair, official_subem, pick_canonical_file
from analysis.compute_pool_acc_crosstab import classify_pool_state, extract_pool_texts

# Source of run outputs: your own fresh runs ("outputs", default) or the shipped
# reference ("reference_outputs"). Override: OUTPUT_ROOT=reference_outputs python ...
OUT = os.environ.get("OUTPUT_ROOT", "outputs")
L4 = ["6k", "32k", "64k", "262k"]
# faithful shared pool (all versions present) -> pure retrieval recall
POOL = (OUT + "/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_struct"
        "/k_100/factconsolidation_sh_{L}/chunksize_512")


def ceiling():
    out = {}
    for L in L4:
        hp = load_haspair(L); qdir = REPO / POOL.format(L=L); b = Counter(); n = 0
        for qid, row in hp.items():
            fs = glob.glob(str(qdir / f"query_{qid}_context_*.json"))
            if not fs:
                continue
            pool = extract_pool_texts(json.load(open(fs[0], encoding="utf-8")), "retrieved_memories")
            b[classify_pool_state(pool, row.get("gt_fact_text") or "", row.get("old_fact_text") or "")] += 1
            n += 1
        out[L] = {"n": n, "new": b["PP-New"] + b["PP-Both"], "both": b["PP-Both"],
                  "old": b["PP-OldOnly"] + b["PP-Both"], "buckets": dict(b)}
    return out


def em_query(rundir):
    out = {}
    for L in L4:
        f = pick_canonical_file(str(REPO / rundir), L); hp = load_haspair(L)
        if not f:
            out[L] = None; continue
        byq = {d["query_id"]: d for d in json.load(open(f))["data"]}
        n = ok = 0
        for qid in hp:
            d = byq.get(qid)
            if d is None:
                continue
            n += 1; ok += bool(official_subem(d.get("output"), d.get("answer")))
        out[L] = (ok, n)
    return out


def em_dontask():
    out = {}
    for L in L4:
        rows = {r["query_id"]: r for r in json.load(
            open(REPO / f"{OUT}/maxserial_theircode/{L}_gpt-4o-mini_vector100.json"))["rows"]}
        hp = load_haspair(L); n = ok = 0
        for qid in hp:
            r = rows.get(qid)
            if r is None:
                continue
            n += 1; ok += bool(official_subem(r.get("answer"), [r.get("gt_answer")]))
        out[L] = (ok, n)
    return out


def pf(t): return f"{100*t[0]/t[1]:.0f}% ({t[0]}/{t[1]})" if t else "—"


def main():
    c = ceiling()
    methods = [
        ("Ours (Struct+LLM-Fallback)", em_query(f"{OUT}/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_no_p5")),
        ("Ours (Struct-Only)",         em_query(f"{OUT}/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_struct")),
        ("Don't Ask",                  em_dontask()),
        ("Vanilla-RAG",                em_query(f"{OUT}/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_q_llm_recency")),
    ]
    print(f"{'':>28} | " + " ".join(f"{L:>12}" for L in L4))
    print(f"{'Retrieval ceiling gt_new∈pool':>28} | " +
          " ".join(f"{100*c[L]['new']/c[L]['n']:>10.1f}% " for L in L4))
    print(f"{'  (both versions∈pool)':>28} | " +
          " ".join(f"{100*c[L]['both']/c[L]['n']:>10.1f}% " for L in L4))
    print("-" * 90)
    for name, d in methods:
        print(f"{name:>28} | " + " ".join(f"{pf(d[L]):>12}" for L in L4))
    print("\nbuckets:", {L: c[L]["buckets"] for L in L4})


if __name__ == "__main__":
    main()
