"""Phase 0 — query-time pipeline (M4 analyze, M5 hybrid retrieve, M6 resolve).

Mirrors guide §5. All structural decisions are pure; the only LLM call is the
null-safe query analyzer (gpt-4o-mini, same model as the triple extractor),
which reuses the SAME DECOMPOSITION_RULES + normalize_* so query (S,P) keys
align with stored (S,P) keys. Misses fall back to the semantic path.

  M4 analyze_query(question)      -> QueryPlan{structural_keys, semantic_query}
  M5 hybrid_retrieve(plan, ...)   -> set(memory_id)  (semantic ∪ structural)
  M6 group_and_resolve(ids, ...)  -> (resolved, ungrouped)
     per (S,P) group: argmax ordinal; KEEP-ALL-ON-TIE. Ordinal is FACT-level
     (per-uid global per-fact counter), so same-(S,P) facts get distinct
     ordinals even within one chunk -> the later-extracted version wins
     (FC freshness). Ties now only arise from genuinely equal ordinals.
  assemble_context(...)           -> str for generation
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

from methods.phase0_triple_extractor import (
    DECOMPOSITION_RULES,
    _llm_chat,
    normalize_predicate,
    normalize_subject,
)

logger = logging.getLogger(__name__)

SEP = "\x1f"  # (S,P) index key separator, matches ingestion

QUERY_ANALYSIS_PROMPT = """\
You are a query analyzer for a memory system. Decide whether the question asks
for the value of a specific (subject, predicate) fact.

If yes, output one or more (subject, predicate) keys such that the question is
asking for the OBJECT (the unknown answer). Use the SAME convention used when
facts were stored:
{rules}

If the question is open-ended (e.g. "how does the user feel lately", "what did
we discuss about X") or cannot be expressed as asking for a specific
(subject, predicate) value, return an empty list. Do NOT force a key.

Question: {question}

Return JSON:
{
  "structural_keys": [{"subject": "...", "predicate": "..."}],
  "semantic_query": "<query rewritten for embedding retrieval, or the original>"
}
""".replace("{rules}", DECOMPOSITION_RULES)


def _qkey(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def analyze_query(question, *, user_id=None, model=None, ollama_url=None, cache_path=None):
    """M4: question -> {structural_keys: [(subject_id, predicate_norm)], semantic_query}.

    Null-safe: open-ended questions yield an empty structural_keys list (Path B
    silently degrades). structural_keys are already normalized to index form.
    Per-question frozen cache (MEM0_QUERY_CACHE) mirrors the triple cache.
    """
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    cache_path = cache_path or os.environ.get("MEM0_QUERY_CACHE")

    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[query-cache] read failed: {e}")

    raw_keys = semantic = None
    ck = _qkey(question)
    if ck in cache:
        raw_keys = cache[ck].get("structural_keys", [])
        semantic = cache[ck].get("semantic_query", question)
    else:
        prompt = QUERY_ANALYSIS_PROMPT.replace("{question}", question)
        try:
            data = json.loads(_llm_chat(prompt, model, ollama_url))
            raw_keys = data.get("structural_keys", []) or []
            semantic = data.get("semantic_query") or question
        except Exception as e:
            logger.error(f"[query-analyze] failed: {e}")
            raw_keys, semantic = [], question
        cache[ck] = {"structural_keys": raw_keys, "semantic_query": semantic}
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = cache_path + ".tmp"
                json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, cache_path)
            except Exception as e:
                logger.error(f"[query-cache] write failed: {e}")

    structural_keys = []
    for k in raw_keys:
        if isinstance(k, dict) and k.get("subject") and k.get("predicate"):
            structural_keys.append(
                (normalize_subject(k["subject"], user_id), normalize_predicate(k["predicate"]))
            )
    return {"structural_keys": structural_keys, "semantic_query": semantic, "raw_keys": raw_keys}


def hybrid_retrieve(plan, memory, sp_index, *, user_id, k=20):
    """M5: union of semantic top-k (Path A) and structural (S,P) lookup (Path B)."""
    cand = set()
    # Path A — semantic (mem0's existing retrieval; embedder = text-embedding-3-small)
    try:
        res = memory.search(query=plan["semantic_query"], user_id=user_id, limit=k)
        hits = res.get("results", res) if isinstance(res, dict) else res
        cand.update(str(h["id"]) for h in hits)
    except Exception as e:
        logger.error(f"[hybrid] semantic path failed: {e}")
    # Path B — structural (no-op if structural_keys empty)
    for s_id, p_norm in plan["structural_keys"]:
        cand.update(str(m) for m in sp_index.get(f"{s_id}{SEP}{p_norm}", []))
    return cand


def group_and_resolve(cand_ids, id2item):
    """M6: group by (subject_id, predicate_norm); per group argmax ordinal,
    KEEP-ALL-ON-TIE. Items without a triple are ungrouped (kept verbatim)."""
    groups, ungrouped = {}, []
    for mid in cand_ids:
        item = id2item.get(str(mid))
        if item is None:
            continue
        tri = (item.get("metadata") or {}).get("triple")
        if tri:
            groups.setdefault((tri["subject_id"], tri["predicate_norm"]), []).append(item)
        else:
            ungrouped.append(item)
    resolved = []
    for items in groups.values():
        mx = max((it.get("metadata") or {}).get("ordinal", -1) for it in items)
        resolved.extend(it for it in items if (it.get("metadata") or {}).get("ordinal", -1) == mx)
    return resolved, ungrouped


def assemble_context(resolved, ungrouped):
    """Newest-version winners + ungrouped, as plain fact lines for generation."""
    lines = []
    for it in resolved + ungrouped:
        txt = it.get("memory") or (it.get("metadata") or {}).get("data", "")
        if txt:
            lines.append(f"- {txt}")
    return "\n".join(lines)
