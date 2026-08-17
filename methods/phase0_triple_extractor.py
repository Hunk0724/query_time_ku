"""Phase 0 — schema-free per-fact (subject, predicate, object) triple extractor.

Role in the framework:
    mem0 L1/L2 extraction yields atomic fact texts. This module turns each
    fact into a (s, p, o) triple so the commit pipeline can build a (S, P)
    inverted index (fact identity) and query-time can group versions of the
    same fact + argmax timestamp. This REPLACES the write-time cross-item
    update-decision LLM call with a cheap, self-contained, per-fact judgment.

Design decisions:
    - MODEL = gpt-4o-mini (default; via MEM0_TRIPLE_MODEL). Of the models tried,
      it was the only one that kept a conflict pair's (S,P) direction consistent
      AND nulled subjective/multi-fact cleanly. It is also the established FC
      inference + memory backbone (no new model introduced). Backend
      auto-selected by name: gpt* -> openai (.env key),
      else ollama (local). Local models remain a no-rate-limit fallback.
    - BATCH = 1 (per-fact; MEM0_TRIPLE_BATCH_SIZE). Multi-fact calls hurt
      reliability (a 7b drops items it extracts fine alone); a frozen cache makes
      call count a one-time cost, so per-fact is the safe default.
    - FROZEN cache — keyed on fact text (timestamp-independent), so SH/MH and
      re-runs produce IDENTICAL triples for free. Mirrors MEM0_EXTRACTION_CACHE.
    - SCOPE = explicit declarative facts only. Subjective state / multi-fact /
      unresolved pronoun -> null. null is a DESIGNED escape hatch (-> semantic
      path only), not a failure; its rate is the F1 metric.
    - NO predicate canonicalization (Phase 0). "lives in" != "resides at".
      Upgrade gated by F2 (§9.1).

Reference: TruthfulRAG (AAAI 2026) — LLM-based relation extraction WITHOUT
    canonicalization (NOT its GraphRAG typed-entity form). (s, p, o) fact
    representation per ROME / MEMIT / MQuAKE / TempReason / CRONQUESTIONS.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

# ── Triple-extraction prompt (generic, NOT task-overfit; see guide §4.2) ────
# Grouping-consistency is the linchpin and is taught by PROSE, not by FC-shaped
# few-shots: subject = the entity the fact is ABOUT and that stays fixed if the
# fact is later updated; object = the value that could change; predicate = the
# relation. This keeps new and old versions of the same fact under the SAME
# (subject, predicate). Examples are deliberately domain-neutral to avoid
# overfitting the extractor to FactConsolidation phrasing.
TRIPLE_EXTRACTION_PROMPT = """\
You are a triple extractor. For EACH input fact (a single declarative
statement extracted from conversation memory), extract its (subject,
predicate, object) triple.

Rules:
- Choose subject = the entity the fact is about, and that stays fixed if the
  fact is later updated; object = the specific value that could change (the
  answer); predicate = the relation linking them. This keeps different
  versions of the same fact under the same (subject, predicate).
- Decompose nominal relations into standard (subject, relation, object) form.
  When the grammatical subject is a noun phrase of the form
  "the <relation> of <entity>" (e.g. "the <ROLE> of <ENTITY>",
  "the <ATTRIBUTE> of <ENTITY>"), do NOT use that whole phrase as the subject
  and do NOT use "is" as the predicate. Instead: subject = <entity>,
  predicate = the relation expressed as a verb phrase (e.g. "has <relation>"),
  object = the stated value. The relation must become the predicate — never be
  swallowed into the subject. This is the standard KG (s, r, o) convention.
- predicate is a SHORT natural-language verb phrase describing the relation.
  Do NOT map it to any fixed vocabulary, snake_case, or canonical form — keep
  the natural wording.
- subject / object are named entities or attribute values, taken verbatim from
  the fact text (do not paraphrase or summarize).
- For pronouns: if the referent is the user/assistant, output "user" or
  "assistant". For other clearly-resolvable pronouns, use the resolved entity
  name. If unresolvable, set triple to null.
- Extract triples regardless of topic, including personal information,
  preferences, opinions, and world knowledge — treat all facts uniformly.
- Never skip or null a fact because it seems factually false. Facts may be
  deliberately counterfactual; extract them anyway and do NOT verify against
  world knowledge. This rule is ONLY about not judging truth — it does NOT
  change how you pick subject/predicate/object; still apply the decomposition
  rules above (e.g. a counterfactual "The R of E is V" still becomes
  subject=E, predicate="has R", object=V).
- If a fact is a subjective state, a complex narrative, OR contains multiple
  independent facts, set its triple to null. Do NOT force a triple.
- null is a valid and ENCOURAGED answer — do not force a triple.
- Confidence reflects how cleanly the fact fits the triple form. Give high
  confidence (>0.8) only when subject, predicate, object are unambiguous.

Examples:
Fact: "The user lives in Taipei."
→ {"subject": "user", "predicate": "lives in", "object": "Taipei", "confidence": 0.95}
Fact: "Marie Curie was born in Warsaw."
→ {"subject": "Marie Curie", "predicate": "was born in", "object": "Warsaw", "confidence": 0.95}
Fact: "The user feels stressed about work lately."
→ null  (subjective state, not cleanly a triple)
Fact: "She moved to Tokyo and got a new job."
→ null  (multiple independent facts + unresolved pronoun)

Input is a numbered list of facts. Return JSON:
{
  "results": [
    {"id": 0, "triple": {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0}},
    {"id": 1, "triple": null}
  ]
}

Facts:
{numbered_facts}
"""


# ── Subject-only extraction (fallback for triple-null memories) ─────────────
# Generalization: even memories that resist a clean (s,p,o) usually have an
# identifiable subject. A subject tag on (almost) every memory lets the query-
# time subject-match guard apply universally, not just to triple-bearing items.
# Domain-neutral: works for personal/conversational AND world-knowledge memory.
SUBJECT_EXTRACTION_PROMPT = """\
Identify the single entity each fact is primarily ABOUT — the subject it would
have if rewritten as "<subject> — <attribute> — <value>".

- About the user themselves: "user".
- About someone/something tied to the user: that specific entity
  (e.g. "user's spouse", "user's son", a named friend, pet, or organization).
  Do NOT collapse these to "user" — they are distinct subjects.
- World knowledge: the named entity the fact is about (the entity, not the role:
  "The capital of France is Paris" -> "France", not "the capital").
Return the subject as a short string; if there is genuinely no identifiable
subject, return null.

Input is a numbered list of facts. Return JSON:
{
  "results": [
    {"id": 0, "subject": "..."},
    {"id": 1, "subject": null}
  ]
}

Facts:
{numbered_facts}
"""


def extract_subjects_batch(
    facts: list[str],
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    cache_path: str | None = None,
    batch_size: int | None = None,
) -> list[str | None]:
    """Extract just the subject entity per fact (fallback for triple-null items).
    Mirrors extract_triples_batch (backend-aware batch + frozen per-fact cache,
    MEM0_SUBJECT_CACHE). Returns raw subject strings (caller normalizes)."""
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    os.environ["MEM0_COST_STAGE"] = "subject_p2b"  # cost-log stage tag
    cache_path = cache_path or os.environ.get("MEM0_SUBJECT_CACHE")
    if batch_size is None:
        _env = os.environ.get("MEM0_TRIPLE_BATCH_SIZE")
        batch_size = int(_env) if _env is not None else (50 if model.lower().startswith("gpt") else 1)
    batch_size = max(1, batch_size)

    cache: dict = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[subject-cache] read failed: {e}")

    results: list[str | None] = [None] * len(facts)
    todo = []
    for i, fact in enumerate(facts):
        c = cache.get(_fact_key(fact))
        if c is not None:
            results[i] = None if (isinstance(c, dict) and c.get("__null__")) else c
        else:
            todo.append((i, fact))

    if todo:
        for start in range(0, len(todo), batch_size):
            sub = todo[start:start + batch_size]
            numbered = "\n".join(f"{j}. {fact}" for j, (_, fact) in enumerate(sub))
            prompt = SUBJECT_EXTRACTION_PROMPT.replace("{numbered_facts}", numbered)
            by_id: dict[int, str | None] = {}
            try:
                data = json.loads(_llm_chat(prompt, model, ollama_url))
                for r in data.get("results", []):
                    if isinstance(r, dict) and "id" in r:
                        s = r.get("subject")
                        by_id[int(r["id"])] = s.strip() if isinstance(s, str) and s.strip() else None
            except Exception as e:
                logger.error(f"[subject-extract] call/parse failed: {e}")
            for j, (orig_idx, fact) in enumerate(sub):
                s = by_id.get(j)
                results[orig_idx] = s
                cache[_fact_key(fact)] = s if s is not None else {"__null__": True}
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = cache_path + ".tmp"
                json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, cache_path)
            except Exception as e:
                logger.error(f"[subject-cache] write failed: {e}")
    return results


# ── Shared decomposition convention (reused by the query analyzer, M4) ──────
# The query-time (S,P) MUST be produced with the SAME subject/predicate
# convention as storage, or keys won't match. This constant is the single
# source of that convention; both the triple prompt (above) and the query
# analysis prompt embed it.
DECOMPOSITION_RULES = """\
- Subject = the entity the fact/question is about, and that stays fixed if the
  fact is later updated; object = the specific value that could change (the
  answer); predicate = the relation linking them.
- Decompose nominal relations into standard (subject, relation, object) form.
  When the phrasing is "the <relation> of <entity>" (e.g. "the author of X",
  "the capital of Y"), do NOT use that whole noun phrase as the subject and do
  NOT use "is" as the predicate. Instead: subject = <entity>, predicate = the
  relation as a verb phrase (e.g. "has <relation>"). The relation must become
  the predicate, never be swallowed into the subject.
- predicate is a SHORT natural-language verb phrase; do NOT map it to a fixed
  vocabulary, snake_case, or canonical form — keep the natural wording.
- subject is a named entity or attribute value, taken verbatim (do not
  paraphrase). For user/assistant referents output "user"/"assistant"."""


# ── Lightweight identity tagging (guide §4.3; pure string, no LLM) ──────────
def normalize_subject(subject_text: str, user_id: str | None = None) -> str:
    # Level-1 ONLY for subjects (entity names): unify hyphens/dashes + collapse
    # whitespace. Deliberately NO article/of stripping — "University of Pisa" /
    # "Bank of America" would be mangled, and stripping "the" would let
    # "the user"-style phrases collapse into the user:: identity.
    s = (subject_text or "").lower().strip()
    s = re.sub(r"[-‐-―]", " ", s)     # L1: hyphens/dashes -> space
    s = re.sub(r"\s+", " ", s).strip()          # L1: collapse whitespace
    if s in {"user", "i", "me", "my", "myself"}:
        return f"user::{user_id}" if user_id else "user"
    if s in {"assistant", "you", "claude", "agent"}:
        return "assistant"
    return re.sub(r"\s+", "_", s)


# Predicate normalization tables (Level-2 = general morphological/function-word
# rules, NOT lexical-semantic mappings -> not overfit to any test set).
_PRED_ARTICLES = {"the", "a", "an", "of"}
_PRED_COPULA = {"is", "was", "are", "were", "be", "been", "being"}


def normalize_predicate(predicate_text: str) -> str:
    s = (predicate_text or "").lower().strip()
    s = re.sub(r"[-_‐-―]", " ", s)        # L1: hyphen/underscore/dash -> space
    toks = re.sub(r"\s+", " ", s).strip().split()   # L1: collapse whitespace
    # L2: drop articles/of; normalize copula/tense (is/was/are/... -> be).
    out = [("be" if t in _PRED_COPULA else t) for t in toks if t not in _PRED_ARTICLES]
    return " ".join(out) or " ".join(toks) or (predicate_text or "").lower().strip()


# ── LLM backends ────────────────────────────────────────────────────────────
# Default = gpt-4o-mini (openai): the strongest instruction follower of the
# candidates tried — only it kept the conflict pair's (S,P)
# direction consistent AND nulled subjective/multi-fact cleanly (qwen2.5:7b was
# close but needed batch=1; mistral:7b produced a 0.95-confidence wrong-direction
# triple that would silently pollute the (S,P) index). gpt-4o-mini is also the
# established FC inference + memory backbone, so no new model is introduced.
# Backend is chosen by model name: "gpt*" -> openai, else ollama (local).
_DOTENV_LOADED = False


def _ensure_env() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        # Load MemoryAgentBench/.env (repo root) without overriding real env vars.
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(_root, ".env"), override=False)
    except Exception:
        pass
    _DOTENV_LOADED = True


def _ollama_chat(prompt: str, model: str, url: str, timeout: int = 600) -> str:
    # num_ctx MUST be set explicitly: ollama's default (~4096) silently truncates
    # the P3 identity-grouping prompt (top-100 candidates ~3.4k tok + JSON output),
    # which made p3_only return {"groups": []} for 100% of queries. Default 8192,
    # override via OLLAMA_NUM_CTX. Applies to all local LLM calls (extraction,
    # triple, grouping, arity) — cheap KV-cache bump, big correctness win.
    _num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
    # num_predict = max output tokens (§M-6 uniform max_tokens bound). Ollama's
    # default is -1 (unbounded up to num_ctx) -> a verbose weak backbone can
    # ramble the whole KV window per extraction/triple/grouping call (llama3.1-8b
    # ingestion measured ~2x slower than it should). Bound it: a 512-tok chunk's
    # (S,P,O) extraction needs <1k tok; P3 grouping over top-100 fits in ~2k.
    # Generous default 2048 (env OLLAMA_NUM_PREDICT); truncation -> that call's
    # step-failure per §M-6, not silent corruption.
    _num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": _num_ctx, "num_predict": _num_predict},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("message", {}).get("content", "")


def _openai_chat(prompt: str, model: str, timeout: int = 120) -> str:
    _ensure_env()
    from openai import OpenAI

    client = OpenAI(timeout=timeout, max_retries=5)  # auto exp-backoff on 429
    import time as _time
    _t0 = _time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    try:  # env-gated cost log (stage set by caller via MEM0_COST_STAGE)
        from methods.cost_logger import log as _costlog
        _u = getattr(resp, "usage", None)
        _costlog(os.environ.get("MEM0_COST_STAGE", "llm_chat"), model,
                 getattr(_u, "prompt_tokens", 0), getattr(_u, "completion_tokens", 0),
                 _time.time() - _t0)
    except Exception:
        pass
    return resp.choices[0].message.content or ""


def _llm_chat(prompt: str, model: str, ollama_url: str) -> str:
    if model.lower().startswith("gpt"):
        return _openai_chat(prompt, model)
    return _ollama_chat(prompt, model, ollama_url)


def _validate_triple(t) -> dict | None:
    """Return a clean triple dict or None. Enforces required string fields."""
    if not isinstance(t, dict):
        return None
    s, p, o = t.get("subject"), t.get("predicate"), t.get("object")
    if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
        return None
    if not (s.strip() and p.strip() and o.strip()):
        return None
    conf = t.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    return {"subject": s.strip(), "predicate": p.strip(), "object": o.strip(),
            "confidence": max(0.0, min(1.0, conf))}


def _fact_key(fact: str) -> str:
    return hashlib.sha256(fact.strip().encode("utf-8")).hexdigest()


def extract_triples_batch(
    facts: list[str],
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    cache_path: str | None = None,
    batch_size: int | None = None,
) -> list[dict | None]:
    """Extract one (s, p, o) triple per fact.

    Returns a list aligned with `facts`; each entry is a validated triple dict
    {subject, predicate, object, confidence} or None (genuine null / parse fail).

    batch_size facts share one LLM call. DEFAULT is backend-aware: 50 for gpt*
    (verified batch=50 -> 0 drops, valid JSON; cuts 455 facts to ~10 calls and
    amortizes the instruction prompt, dodging rate/token limits), 1 for local
    7b (which drops/nulls items in multi-fact calls it extracts fine alone).
    Override via MEM0_TRIPLE_BATCH_SIZE.

    Caching: per-fact, keyed on fact text. A cached entry is either a triple
    dict or the sentinel {"__null__": true}. Only uncached facts hit the model.
    """
    os.environ["MEM0_COST_STAGE"] = "triple_p2"  # cost-log stage tag
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    cache_path = cache_path or os.environ.get("MEM0_TRIPLE_CACHE")
    if batch_size is None:
        _env_bs = os.environ.get("MEM0_TRIPLE_BATCH_SIZE")
        if _env_bs is not None:
            batch_size = int(_env_bs)
        else:
            # gpt-4o-mini batches reliably (verified batch=50: 0 drops, valid
            # JSON) — batch to dodge rate limits / amortize the instruction
            # prompt. Local 7b drops items in multi-fact calls -> stay per-fact.
            batch_size = 50 if model.lower().startswith("gpt") else 1
    batch_size = max(1, batch_size)

    cache: dict = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            logger.error(f"[triple-cache] read failed: {e}")

    results: list[dict | None] = [None] * len(facts)
    todo = []  # (orig_idx, fact)
    for i, fact in enumerate(facts):
        c = cache.get(_fact_key(fact))
        if c is not None:
            results[i] = None if (isinstance(c, dict) and c.get("__null__")) else c
        else:
            todo.append((i, fact))

    if todo:
        for start in range(0, len(todo), batch_size):
            sub = todo[start:start + batch_size]
            numbered = "\n".join(f"{j}. {fact}" for j, (_, fact) in enumerate(sub))
            prompt = TRIPLE_EXTRACTION_PROMPT.replace("{numbered_facts}", numbered)
            parsed_by_id: dict[int, dict | None] = {}
            try:
                raw = _llm_chat(prompt, model, ollama_url)
                data = json.loads(raw)
                for r in data.get("results", []):
                    if isinstance(r, dict) and "id" in r:
                        parsed_by_id[int(r["id"])] = _validate_triple(r.get("triple"))
            except Exception as e:
                logger.error(f"[triple-extract] call/parse failed: {e}")

            for j, (orig_idx, fact) in enumerate(sub):
                triple = parsed_by_id.get(j)  # None if missing or invalid
                results[orig_idx] = triple
                cache[_fact_key(fact)] = triple if triple is not None else {"__null__": True}

        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = cache_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                os.replace(tmp, cache_path)
            except Exception as e:
                logger.error(f"[triple-cache] write failed: {e}")

    return results


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        "goaltender is associated with the sport of pesäpallo.",
        "The author of Our Mutual Friend is Charles Dickens.",
        "The author of Our Mutual Friend is Charles Darwin.",
        "The user lives in Taipei.",
        "The user feels stressed about work lately.",
        "She moved to Tokyo and got a new job.",
    ]
    out = extract_triples_batch(sample, cache_path="/tmp/phase0_triple_smoke_cache.json")
    n_ok = n_null = 0
    for fact, t in zip(sample, out):
        if t is None:
            n_null += 1
            print(f"  NULL  | {fact}")
        else:
            n_ok += 1
            sid = normalize_subject(t["subject"], user_id="smoke")
            pn = normalize_predicate(t["predicate"])
            print(f"  OK c={t['confidence']:.2f} | ({sid} | {pn} | {t['object']}) | {fact}")
    print(f"\nF1(null rate) = {n_null}/{len(sample)} = {n_null/len(sample):.0%}  | ok={n_ok}")
