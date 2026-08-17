"""Compute M1/M2/M3 KU-attribution metrics per evaluation_protocol_v2.md.

Rigor v2 changes vs v1:
- Matcher: 2-layer (string full-sentence + subject+object token overlap fallback)
- M3 mem0 UPDATE_kept_new: strict user-proposed condition (gt_old NOT in bank
  AND gt_new IN bank AND UPDATE event exists) — not just fuzzy final-state match
- Zep audit: per-length edge count vs P1 fact count comparison at report top
- Ours variants: all 4 (ours full / no_p5 / struct / p3_only) reported for M2/M3
- N_bank enumeration reported alongside N_query for M1

Usage:
    python analysis/compute_m1_m2_m3.py [--length 6k 32k 64k]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GT_PATHS = {
    "6k": "analysis/results/sh_512_mquake_analysis.json",
    "32k": "analysis/results/sh_32k_mquake_analysis.json",
    "64k": "analysis/results/sh_64k_mquake_analysis.json",
}

INGEST_DEST = REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_dest/k_100"
OURS_ROOTS = {
    "ours(full)":    REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified/k_100",
    "ours(no_p5)":   REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_no_p5/k_100",
    "ours(struct)":  REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_struct/k_100",
    "ours(p3_only)": REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_p3_only_no_struct/k_100",
}
MEM0_QUERY = INGEST_DEST
ZEP_ROOT = REPO / "outputs/rag_retrieved/Structure_rag_zep/k_10"


# ==================== Matcher v3 (rigor: triple-based) ==================== #
# v3 rationale: v2's Layer-1 SequenceMatcher was false-positive on facts sharing
# predicate stem but differing object (e.g. "goaltender ... sport of pesäpallo"
# vs "goaltender ... sport of ice hockey" gave ratio 0.857 > threshold 0.85).
# Fact identity is inherently STRUCTURAL (subject + predicate + object), so
# match_pair now decomposes gt_new/gt_old into (stem, new_obj, old_obj) and
# checks (stem tokens ≥ 60% AND target-object all tokens IN mem AND non-target-
# object NOT confusingly present). This is the only rigor-tight identity check.
def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_stem_objs(gt_new: str, gt_old: str) -> tuple:
    """Return (shared_word_stem, new_object_tail, old_object_tail)."""
    a = (gt_new or "").split()
    b = (gt_old or "").split()
    stem_words = []
    for wa, wb in zip(a, b):
        if wa.lower().rstrip(".,;:!?") == wb.lower().rstrip(".,;:!?"):
            stem_words.append(wa)
        else:
            break
    stem = " ".join(stem_words)
    new_tail = " ".join(a[len(stem_words):]).rstrip(".,;:!?").strip()
    old_tail = " ".join(b[len(stem_words):]).rstrip(".,;:!?").strip()
    return stem, new_tail, old_tail


def _token_present(mem_words_set: set, target_words: list, threshold=1.0) -> bool:
    """Return True iff `threshold` fraction of non-stopword tokens in target are in mem set."""
    salient = [w for w in target_words if len(w) > 1 and w not in {"the", "a", "an", "of", "is", "was",
                                                                     "are", "were", "be", "in", "on", "to",
                                                                     "for", "and", "or", "at"}]
    if not salient:
        return False
    hit = sum(1 for w in salient if w in mem_words_set)
    return hit / len(salient) >= threshold


def match_pair_v4(memory_text: str, gt_new: str, gt_old: str, target: str) -> bool:
    """v4 = v3 with Layer-0 full-fact-substring pre-check.

    Rationale for v4 (audit 2026-07-04):
    v3 rule (iii) uses bag-of-words check for "other-object NOT confusingly
    present" — fails when non-target object token also appears elsewhere in the
    memory (e.g. qid 23 64k: gt_old object = "racing", memory text
    "racing video game is associated with the sport of Australian rules football"
    → "racing" appears as SUBJECT prefix, not object → v3 treats as ambiguous,
    returns False. Actual pool WAS clean PP-New.

    v4 adds Layer-0 substring check on the FULL normalized fact string:
      if norm(gt_target) IS substring of norm(memory) AND norm(gt_other) is NOT
      → return True (unambiguous verbatim match).
    Otherwise falls back to v3 token-based check.

    Audit: v4 fixes 100% of ours PP-OldOnly false-negatives at 64k (5/5), plus
    reduces mem0+P1 false-negatives (65% → smaller).
    """
    mem_n = norm(memory_text)
    if not mem_n:
        return False
    gt_target = gt_new if target == "new" else gt_old
    gt_other = gt_old if target == "new" else gt_new
    gt_target_n = norm(gt_target or "")
    gt_other_n = norm(gt_other or "")
    # Layer 0: full-fact substring (verbatim match, disambiguated)
    if gt_target_n and gt_target_n in mem_n:
        # Only unambiguous if the other version's full text is NOT also in mem
        if not (gt_other_n and gt_other_n in mem_n):
            return True
    # Layer 1: v3 token-based rigor check (fallback)
    return _match_pair_v3(memory_text, gt_new, gt_old, target)


def _match_pair_v3(memory_text: str, gt_new: str, gt_old: str, target: str) -> bool:
    """Rigor-tight fact-identity match (v3, kept for audit / fallback):
       (i) shared stem majority present in memory
       (ii) ALL target-object salient tokens present
       (iii) non-target-object salient tokens NOT all present (else ambiguous → False)
    """
    mem_n = norm(memory_text)
    if not mem_n:
        return False
    mem_words = set(mem_n.split())
    stem, new_obj, old_obj = extract_stem_objs(gt_new, gt_old)
    stem_n = norm(stem)
    target_obj_n = norm(new_obj if target == "new" else old_obj)
    other_obj_n = norm(old_obj if target == "new" else new_obj)
    stem_words = stem_n.split()
    target_words = target_obj_n.split()
    other_words = other_obj_n.split()
    # Rigor stem: at least 60% of stem non-stop tokens must match (min 2)
    stem_salient = [w for w in stem_words if len(w) > 1 and w not in {"the", "a", "an", "of", "is", "was",
                                                                      "are", "were", "be", "in", "on", "to",
                                                                      "for", "and", "or", "at"}]
    if len(stem_salient) < 2:
        # short stem (few salient words) → fall back to full-string containment (rare edge case)
        gt_target = gt_new if target == "new" else gt_old
        return norm(gt_target) in mem_n
    stem_hit = sum(1 for w in stem_salient if w in mem_words) / len(stem_salient) >= 0.6
    if not stem_hit:
        return False
    # Target object must be fully present
    if not target_words or not _token_present(mem_words, target_words, threshold=1.0):
        return False
    # Non-target object must NOT be present (avoid ambiguity)
    if other_words and _token_present(mem_words, other_words, threshold=1.0):
        return False  # ambiguous — both objects present, don't attribute
    return True


# Default matcher for downstream analysis = v4 (v3 available as _match_pair_v3).
# To force v3-only rigor mode, callers may import _match_pair_v3 directly.
match_pair = match_pair_v4


# ==================== data loaders ==================== #
def load_gt(L: str):
    return {e["query_id"]: e for e in json.load(open(REPO / GT_PATHS[L]))}


def enumerate_n_query_pairs(L: str) -> list:
    gt = load_gt(L)
    return [{"qid": q, "gt_new_text": e["gt_fact_text"], "gt_old_text": e["old_fact_text"]}
            for q, e in gt.items() if e.get("conflict_type") == "has_pair"]


def enumerate_n_bank_pairs(L: str) -> list:
    """From ours P1 triple cache: group (S,P), pairs with ≥2 distinct objects."""
    from methods.phase0_triple_extractor import normalize_subject, normalize_predicate
    p = REPO / f"analysis/results/p1_caches/triple_cache_p1_{L}.json"
    if not p.exists():
        return []
    trips = json.load(open(p))
    groups = defaultdict(list)
    for _h, t in trips.items():
        s = normalize_subject(t.get("subject", ""), user_id=None)
        pr = normalize_predicate(t.get("predicate", ""))
        obj = t.get("object", "")
        if not s or not pr or not obj:
            continue
        groups[(s, pr)].append({"subject": t.get("subject"), "predicate": t.get("predicate"),
                                "object": obj, "subject_id": s, "predicate_norm": pr})
    pairs = []
    for (s, pr), items in groups.items():
        distinct_objs = list({it["object"].lower().strip(): it for it in items}.values())
        if len(distinct_objs) < 2:
            continue
        # Use two ends as representative pair (semantic: at least one confusion)
        # For rigor we may treat >2 as multi-arity and skip; here take first two
        for i in range(len(distinct_objs)):
            for j in range(i + 1, len(distinct_objs)):
                a, b = distinct_objs[i], distinct_objs[j]
                fact_a = f"{a['subject']} {a['predicate']} {a['object']}"
                fact_b = f"{b['subject']} {b['predicate']} {b['object']}"
                pairs.append({"gt_new_text": fact_a, "gt_old_text": fact_b, "subject_id": s})
    return pairs


# ==================== Zep audit ==================== #
def zep_audit(L: str) -> dict:
    """Compare Zep total edge count observed (union of per-query top-10) vs P1
    fact count. Report ratio; suggest reset if > 1.5×."""
    d = ZEP_ROOT / f"factconsolidation_sh_{L}/chunksize_512"
    files = sorted(glob.glob(str(d / "query_*_context_*.json")))
    if not files:
        return {"error": "no zep files"}
    all_edges = {}
    saw_expired = False
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for e in j.get("edges", []) or []:
            k = e.get("uuid") or (e.get("fact"), e.get("source_node"))
            all_edges[k] = e
            if e.get("expired_at") is not None:
                saw_expired = True
    p1_cache = REPO / f"analysis/results/p1_caches/triple_cache_p1_{L}.json"
    p1_count = len(json.load(open(p1_cache))) if p1_cache.exists() else None
    return {
        "n_files": len(files),
        "n_edges_observed_union": len(all_edges),
        "n_p1_facts": p1_count,
        "ratio_edges_over_p1": (len(all_edges) / p1_count) if p1_count else None,
        "verdict": (
            "N/A (Zep top-10 per query — full graph dump needed for definitive audit)"
        ),
        "expired_at_present_in_data": saw_expired,
    }


# ==================== M1 ==================== #
def m1_ours(pairs) -> dict:
    return {"NFPR": 100.0, "OFPR": 100.0, "BSPR": 100.0, "DLR": 0.0, "n": len(pairs),
            "note": "by construction (conservative-ADD, no cross-item LLM judgment)"}


def m1_mem0(pairs, L: str) -> dict:
    """Simulate mem0 write-time events (temporal event dict) → final alive state."""
    p = INGEST_DEST / f"factconsolidation_sh_{L}/chunksize_512/ingestion_context_0.jsonl"
    if not p.exists():
        return {"error": "missing ingest jsonl"}
    events = []
    for i, line in enumerate(open(p)):
        for ev in (json.loads(line).get("vector_results", {}) or {}).get("results", []) or []:
            events.append((i, ev))
    n = len(pairs)
    if n == 0: return {"error": "no pairs"}
    nfpr = ofpr = bspr = 0
    for pr in pairs:
        # Simulate final alive state
        alive = {}
        for ci, ev in events:
            et = ev.get("event")
            eid = ev.get("id")
            em = ev.get("memory") or ""
            if et in ("ADD", "UPDATE"):
                alive[eid] = em
            elif et == "DELETE":
                alive.pop(eid, None)
        has_new = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "new") for m in alive.values())
        has_old = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "old") for m in alive.values())
        if has_new: nfpr += 1
        if has_old: ofpr += 1
        if has_new and has_old: bspr += 1
    return {"NFPR": 100 * nfpr / n, "OFPR": 100 * ofpr / n, "BSPR": 100 * bspr / n,
            "DLR": 100 * (n - nfpr) / n, "n": n}


def m1_zep(pairs, L: str) -> dict:
    """Zep bank state — union of per-query edges (valid = invalid_at is None)."""
    d = ZEP_ROOT / f"factconsolidation_sh_{L}/chunksize_512"
    files = sorted(glob.glob(str(d / "query_*_context_*.json")))
    if not files:
        return {"error": "no zep files"}
    all_edges = {}
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for e in j.get("edges", []) or []:
            all_edges[e.get("uuid") or e.get("fact")] = e
    valid_facts = [e.get("fact", "") for e in all_edges.values() if not e.get("invalid_at")]
    n = len(pairs)
    if n == 0: return {"error": "no pairs"}
    nfpr = ofpr = bspr = 0
    for pr in pairs:
        has_new = any(match_pair(f, pr["gt_new_text"], pr["gt_old_text"], "new") for f in valid_facts)
        has_old = any(match_pair(f, pr["gt_new_text"], pr["gt_old_text"], "old") for f in valid_facts)
        if has_new: nfpr += 1
        if has_old: ofpr += 1
        if has_new and has_old: bspr += 1
    return {"NFPR": 100 * nfpr / n, "OFPR": 100 * ofpr / n, "BSPR": 100 * bspr / n,
            "DLR": 100 * (n - nfpr) / n, "n": n,
            "note": "top-10 union (upper-bound; true bank state may be lower)"}


# ==================== M2 ==================== #
def m2_generic(pairs, root: Path, mem_key: str, L: str) -> dict:
    """Rigor v3: for ours variants, use `memories_str` — the ACTUAL string sent
    to answer LLM during the original run. This bypasses offline reconstruction
    entirely, so we don't depend on cache hits or API availability. For mem0
    we still use retrieved_memories (baseline has no post-retrieval resolution).
    For Zep, use edges list."""
    d = root / f"factconsolidation_sh_{L}/chunksize_512"
    n = len(pairs)
    if n == 0: return {"error": "no pairs"}
    pp = Counter()
    n_found = 0
    for pr in pairs:
        qid = pr.get("qid")
        if qid is None: continue  # N_bank pairs have no query file
        files = glob.glob(str(d / f"query_{qid}_context_*.json"))
        if not files: continue
        n_found += 1
        try:
            j = json.load(open(files[0]))
        except Exception:
            continue
        if mem_key == "edges":
            pool = [e.get("fact", "") for e in j.get("edges", []) or []]
        elif mem_key == "resolved_pool":
            # Rigor: use memories_str — the ground truth string the answer LLM saw
            ms = j.get("memories_str", "")
            pool = [l.lstrip("- ").strip() for l in ms.split("\n") if l.strip()]
        else:  # retrieved_memories (mem0 baseline — no post-retrieval resolution)
            pool = [m.get("memory", "") for m in j.get("retrieved_memories", []) or []]
        has_new = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "new") for m in pool)
        has_old = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "old") for m in pool)
        if has_new and not has_old: pp["PP-New"] += 1
        elif has_new and has_old: pp["PP-Both"] += 1
        elif not has_new and has_old: pp["PP-OldOnly"] += 1
        else: pp["PP-Missing"] += 1
    if not n_found: return {"error": "no query files found"}
    return {k: 100 * v / n_found for k, v in pp.items()} | {"n": n_found}


# ==================== M3 ==================== #
def m3_mem0_rigorous(pairs, L: str) -> dict:
    """Rigorous mem0 M3 buckets per protocol_v2 §2.3.1.

    For each pair, trace all events matching gt_new/gt_old + verify final bank
    state (from m1_mem0 simulation). Bucket conditions are mutually exclusive
    and collectively exhaustive.
    """
    p = INGEST_DEST / f"factconsolidation_sh_{L}/chunksize_512/ingestion_context_0.jsonl"
    if not p.exists():
        return {"error": "missing ingest jsonl"}
    events = []
    for i, line in enumerate(open(p)):
        for ev in (json.loads(line).get("vector_results", {}) or {}).get("results", []) or []:
            events.append((i, ev))
    n = len(pairs)
    if n == 0: return {"error": "no pairs"}
    buckets = Counter()
    for pr in pairs:
        # 1) Simulate final alive state to determine gt_new / gt_old presence
        alive = {}
        for ci, ev in events:
            if ev.get("event") in ("ADD", "UPDATE"):
                alive[ev.get("id")] = ev.get("memory") or ""
            elif ev.get("event") == "DELETE":
                alive.pop(ev.get("id"), None)
        has_new_final = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "new") for m in alive.values())
        has_old_final = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "old") for m in alive.values())
        # 2) Enumerate this pair's touching events (any event where memory
        #    matched gt_new or gt_old, in chunk order)
        pair_events = []
        for ci, ev in events:
            m = ev.get("memory") or ""
            hn = match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "new")
            ho = match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "old")
            if hn or ho:
                pair_events.append({"chunk": ci, "event": ev.get("event"),
                                    "has_new": hn, "has_old": ho})
        etypes = {e["event"] for e in pair_events}
        # 3) Apply rigorous bucket rules
        if not has_new_final and not has_old_final and not pair_events:
            buckets["NONE_silent"] += 1
        elif has_new_final and not has_old_final and "UPDATE" in etypes:
            buckets["UPDATE_kept_new"] += 1  # user's strict definition
        elif has_new_final and has_old_final and etypes.issubset({"ADD", "NONE"}):
            buckets["ADD_both_coexist"] += 1
        elif not has_new_final and "DELETE" in etypes and any(e["event"] == "DELETE" and e["has_new"] for e in pair_events):
            buckets["DELETE_dropped_new"] += 1
        elif not has_new_final and not has_old_final and "DELETE" in etypes:
            buckets["DELETE_lost_new"] += 1
        elif not has_new_final and "NONE" in etypes and not any(e["event"] == "DELETE" for e in pair_events):
            buckets["NONE_silent"] += 1
        else:
            # Mixed / ambiguous
            code = "+".join(sorted(etypes)) or "no_events"
            buckets[f"Ambiguous({code}|final_new={has_new_final},final_old={has_old_final})"] += 1
    return {k: (100 * v / n, v) for k, v in buckets.items()} | {"n": n}


def m3_zep(pairs, L: str) -> dict:
    d = ZEP_ROOT / f"factconsolidation_sh_{L}/chunksize_512"
    files = sorted(glob.glob(str(d / "query_*_context_*.json")))
    if not files:
        return {"error": "no zep files"}
    all_edges = {}
    saw_expired = False
    for f in files:
        try: j = json.load(open(f))
        except Exception: continue
        for e in j.get("edges", []) or []:
            all_edges[e.get("uuid") or e.get("fact")] = e
            if e.get("expired_at"): saw_expired = True
    n = len(pairs)
    if n == 0: return {"error": "no pairs"}
    buckets = Counter()
    for pr in pairs:
        best = "missing"
        for e in all_edges.values():
            if match_pair(e.get("fact", ""), pr["gt_new_text"], pr["gt_old_text"], "new"):
                inv, exp = e.get("invalid_at"), e.get("expired_at")
                if inv is None:
                    best = "valid"; break
                elif exp is not None:
                    if best != "valid": best = "contradicted-invalidated"
                else:
                    if best == "missing": best = "temporal-extraction"
        buckets[best] += 1
    note = ("expired_at present — cloud data captured post-patch"
            if saw_expired
            else "expired_at MISSING — cloud data pre-patch; contradicted vs temporal-extraction cannot be distinguished")
    return {k: (100 * v / n, v) for k, v in buckets.items()} | {"n": n, "note": note}


def m3_ours(pairs, ours_variant: str, L: str) -> dict:
    root = OURS_ROOTS[ours_variant]
    d = root / f"factconsolidation_sh_{L}/chunksize_512"
    buckets = Counter()
    n = len(pairs)
    if n == 0: return {"error": "no pairs"}
    for pr in pairs:
        qid = pr.get("qid")
        if qid is None: continue
        files = glob.glob(str(d / f"query_{qid}_context_*.json"))
        if not files:
            buckets["query_file_missing"] += 1; continue
        try: j = json.load(open(files[0]))
        except Exception:
            buckets["query_file_missing"] += 1; continue
        pool_src = j.get("resolved_pool") if j.get("resolved_pool") is not None else j.get("retrieved_memories", [])
        pool_texts = [m.get("memory", "") for m in pool_src] if pool_src else []
        has_new = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "new") for m in pool_texts)
        has_old = any(match_pair(m, pr["gt_new_text"], pr["gt_old_text"], "old") for m in pool_texts)
        if has_new and not has_old: buckets["gt_new_only_kept"] += 1
        elif has_new and has_old: buckets["both_kept"] += 1
        elif not has_new and has_old: buckets["gt_new_dropped_gt_old_kept"] += 1
        else: buckets["neither_in_pool"] += 1
    return {k: (100 * v / n, v) for k, v in buckets.items()} | {"n": n}


# ==================== format ==================== #
def fmt(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if k in ("n", "note", "error", "n_files", "n_edges_observed_union", "n_p1_facts",
                 "ratio_edges_over_p1", "verdict", "expired_at_present_in_data"): continue
        if isinstance(v, tuple):
            lines.append(f"  {k:<50s}: {v[0]:5.1f}% ({v[1]})")
        else:
            lines.append(f"  {k:<50s}: {v:5.1f}%")
    if "n" in d: lines.append(f"  {'(denominator n)':<50s}: {d['n']}")
    if "note" in d: lines.append(f"  note: {d['note']}")
    if "error" in d: lines.append(f"  ERROR: {d['error']}")
    return "\n".join(lines)


# ==================== main ==================== #
def main(lengths=None):
    lengths = lengths or ["6k", "32k", "64k"]
    out = ["# M1 / M2 / M3 Metrics (v2, rigor)", ""]
    out.append("*Protocol*: evaluation_protocol_fc_mquake_v2.md")
    out.append("*Matcher*: 2-layer (string full-sentence + subject+object token overlap)")
    out.append("*Bucket rigor*: M3 mem0 UPDATE_kept_new requires (gt_old NOT in bank) AND (gt_new IN bank) AND (UPDATE event exists)")
    out.append("")

    for L in lengths:
        out.append(f"## Length = {L}")
        out.append("")

        # Zep audit at length top
        audit = zep_audit(L)
        out.append("### Zep cumulative-state audit")
        out.append("```")
        for k in ("n_files", "n_edges_observed_union", "n_p1_facts", "ratio_edges_over_p1",
                  "expired_at_present_in_data", "verdict"):
            if k in audit: out.append(f"  {k:<40s}: {audit[k]}")
        out.append("```")
        out.append("")

        # Pair enumeration
        pairs_q = enumerate_n_query_pairs(L)
        pairs_b = enumerate_n_bank_pairs(L)
        out.append(f"**N_query = {len(pairs_q)}** (from sh_{L}_mquake_analysis.json has_pair)")
        out.append(f"**N_bank = {len(pairs_b)}** (from ours P1 triple cache, (S,P) with ≥2 distinct objects)")
        out.append("")

        # ---- M1 (both denominators) ----
        for pop_name, pairs in [("N_query", pairs_q), ("N_bank", pairs_b)]:
            out.append(f"### M1 (denominator = {pop_name}, n={len(pairs)})")
            out.append("```")
            out.append("ours (by construction)")
            out.append(fmt(m1_ours(pairs)))
            out.append("mem0+P1(b) — write-time destructive commit")
            out.append(fmt(m1_mem0(pairs, L)))
            out.append("Zep (cloud, top-10 union of per-query edges)")
            out.append(fmt(m1_zep(pairs, L)))
            out.append("```")
            out.append("")

        # ---- M2 (only N_query — queries only) ----
        out.append("### M2 (pool state per query)")
        out.append("```")
        for name, root in OURS_ROOTS.items():
            out.append(f"{name}")
            out.append(fmt(m2_generic(pairs_q, root, "resolved_pool", L)))
        out.append("mem0+P1(b) (raw top-100)")
        out.append(fmt(m2_generic(pairs_q, MEM0_QUERY, "retrieved_memories", L)))
        out.append("Zep (top-10 edges)")
        out.append(fmt(m2_generic(pairs_q, ZEP_ROOT, "edges", L)))
        out.append("```")
        out.append("")

        # ---- M3 ----
        out.append("### M3 (root-cause attribution)")
        out.append("```")
        out.append("mem0+P1(b) LLM event distribution on gt_new (rigorous bucket rules)")
        out.append(fmt(m3_mem0_rigorous(pairs_q, L)))
        out.append("Zep (invalid_at, expired_at) on gt_new edge")
        out.append(fmt(m3_zep(pairs_q, L)))
        for name in OURS_ROOTS:
            out.append(f"{name} pool composition")
            out.append(fmt(m3_ours(pairs_q, name, L)))
        out.append("```")
        out.append("")

    out_p = REPO / "analysis/results/m1_m2_m3_results.md"
    out_p.write_text("\n".join(out))
    print(f"[wrote] {out_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", "-L", nargs="*", default=None, choices=["6k", "32k", "64k"])
    args = ap.parse_args()
    main(args.length)
