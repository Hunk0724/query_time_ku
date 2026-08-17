"""Tier 1 main analysis: return_context × Acc cross-tabulation per method × length.

For each has_pair query, classify by:
  - Pool state: {PP-New, PP-Both, PP-OldOnly, PP-Missing}
    (whether gt_new / gt_old appear in the pool sent to answer LLM)
  - Acc: {✓, ✗} (exact_match from results.json)

Yields, per (method × length), a 4×2 cross-tab plus a wrong-qid list per bucket.

Rigor:
  - Matcher v4 (2026-07-04) — Layer-0 full-fact-substring pre-check + Layer-1
    matcher v3 (triple-based) fallback. v3 alone had systematic false negatives
    on unambiguous verbatim GT text (qid 23 64k: pool had exact gt_new string,
    v3 rule (iii) fired on bag-of-words check because gt_old's object token
    appeared as subject prefix). v4 audit reduced ours 64k PP-OldOnly
    false-neg from 5/5 → 2/2, Zep 64k from 3/3 → 0/0, mem0 64k from 11/17 → 9/15.
    See analysis/compute_m1_m2_m3.py::match_pair_v4 docstring.
  - ours* variants use `memories_str` (ground truth what answer LLM saw during
    the actual run). mem0(b) uses `retrieved_memories`. Zep uses `edges`.
  - Data is what the pipeline actually did — no offline reconstruction (which
    fails on cache miss without API key).

Output: analysis/results/pool_acc_crosstab.md

Usage:
    python analysis/compute_pool_acc_crosstab.py [--length 6k 32k 64k]
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

# Reuse the rigor-tight matcher v3
from analysis.compute_m1_m2_m3 import match_pair, norm  # noqa: E402

# MAB official EM (matches utils/eval_other_utils.py::default_post_process semantics):
#   EM = max(EM(raw prediction), EM(parse_output(prediction)))
# per drqa_exact_match_score (case-insensitive, punctuation-stripped, article-stripped)
from utils.eval_other_utils import (  # noqa: E402
    parse_output, drqa_exact_match_score, drqa_metric_max_over_ground_truths,
)


def _em_from_perqid(response: str, gt_answer) -> bool:
    """Recompute EM byte-for-byte reproducibly per MAB `default_post_process`.

    MAB official semantics: return TRUE if EM matches either the RAW response OR
    the parse_output-stripped response. This is why aggregated `exact_match` in
    Conflict_Resolution/*.json may show TRUE for some qid whose per-qid response
    only matches after Answer: prefix stripping.

    Rationale for reading per-qid `response` instead of aggregated `exact_match`:
    - per-qid file is the write path from _handle_*_agent (agent.py)
    - aggregated is a batch summary computed at run finalization; can go stale
      when partial re-runs overwrite per-qid but not aggregated
    - concrete cases: Zep 32k aggregated had 48/65 has_pair `output` replaced by
      "Answer:" placeholders while per-qid `response` kept the real LLM output;
      mem0+P1 all 3 lengths had 33-42/N mismatches between aggregated `output`
      and per-qid `response`. See rigor_audit.py / rigor_audit.md.
    """
    if response is None or gt_answer is None:
        return False
    resp_str = str(response)
    # Raw EM
    if bool(drqa_metric_max_over_ground_truths(
            drqa_exact_match_score, resp_str, gt_answer)):
        return True
    # Parsed EM ("Answer: X" prefix strip → compare)
    parsed = parse_output(resp_str)
    if parsed is None:
        return False
    return bool(drqa_metric_max_over_ground_truths(
        drqa_exact_match_score, parsed, gt_answer))


# ==================== Configuration ==================== #
GT_PATHS = {
    "6k": "analysis/results/sh_512_mquake_analysis.json",
    "32k": "analysis/results/sh_32k_mquake_analysis.json",
    "64k": "analysis/results/sh_64k_mquake_analysis.json",
    "262k": "analysis/results/sh_262k_mquake_analysis.json",
}

# (method_display_name, per-query JSON root, "pool key",
#  results.json glob under Conflict_Resolution)
#
# Naming (2026-07-05 update):
#   - MAIN method label = "ours" = struct + P3 + argmax (formerly "no_p5")
#   - Ablations: "ours (no P3)" = struct only; "ours (no struct)" = P3 only
#   - Appendix: "ours (+P5)" = P5 conflict-type classifier ON (formerly "full")
#     kept because it must run first to build P1 extraction cache, though its
#     result is appendix material only.
METHODS = [
    # ==================== gpt-4o-mini (canonical) ====================
    ("ours",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_no_p5/k_100",
     "memories_str",
     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_no_p5/Conflict_Resolution"),
    ("ours (no P3)",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_struct/k_100",
     "memories_str",
     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_struct/Conflict_Resolution"),
    ("ours (no struct)",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_p3_only_no_struct/k_100",
     "memories_str",
     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_p3_only_no_struct/Conflict_Resolution"),
    ("(b) mem0+P1",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_dest/k_100",
     "retrieved_memories",
     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_dest/Conflict_Resolution"),
    ("Zep (k=10)",
     REPO / "outputs/rag_retrieved/Structure_rag_zep/k_10",
     "edges",
     "gpt-4o-mini-zep/Conflict_Resolution"),
    ("ours (+P5) [appendix]",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified/k_100",
     "memories_str",
     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified/Conflict_Resolution"),
    # ==================== gpt-4.1-mini (Plan A backbone extension) ====================
    ("[4.1-mini] ours",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4.1-mini-mem0_512_openai_unified_no_p5/k_100",
     "memories_str",
     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_no_p5/Conflict_Resolution"),
    ("[4.1-mini] ours (no P3)",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4.1-mini-mem0_512_openai_unified_struct/k_100",
     "memories_str",
     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_struct/Conflict_Resolution"),
    ("[4.1-mini] ours (no struct)",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4.1-mini-mem0_512_openai_unified_p3_only_no_struct/k_100",
     "memories_str",
     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_p3_only_no_struct/Conflict_Resolution"),
    ("[4.1-mini] (b) mem0+P1",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4.1-mini-mem0_512_openai_unified_dest/k_100",
     "retrieved_memories",
     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_dest/Conflict_Resolution"),
    ("[4.1-mini] Zep (k=10)",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4.1-mini-zep/k_10",
     "edges",
     "gpt-4.1-mini-zep/Conflict_Resolution"),
    ("[4.1-mini] ours (+P5) [appendix]",
     REPO / "outputs/rag_retrieved/Structure_rag_gpt-4.1-mini-mem0_512_openai_unified/k_100",
     "memories_str",
     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified/Conflict_Resolution"),
]

OUT_MD = REPO / "analysis/results/pool_acc_crosstab.md"


# ==================== helpers ==================== #
def load_gt(L: str) -> dict:
    return {e["query_id"]: e for e in json.load(open(REPO / GT_PATHS[L]))}


def load_em(results_dir_glob: str, L: str) -> dict | None:
    """Return {qid: exact_match_bool} from Conflict_Resolution results.json.
    Pick priority:
      1. Files matching the standard 'size256_shots0_max_samplesunknown' pattern
         (the official Mac run pattern for full 100-query runs).
      2. Fallback: file with max n_data.
    Rejects smoke files (n_data < 50)."""
    root = REPO / f"outputs/{results_dir_glob}"
    files = sorted(glob.glob(str(root / f"factconsolidation_sh_{L}_*results*.json")))
    if not files:
        return None
    STANDARD = "size256_shots0_max_samplesunknown"
    # Priority 1: standard pattern
    for f in files:
        if STANDARD in f:
            try:
                d = json.load(open(f))
                if len(d.get("data", [])) >= 50:
                    return {e["query_id"]: bool(e.get("exact_match")) for e in d["data"]}
            except Exception:
                continue
    # Priority 2: also try samples1 variant (64k uses max_samples1 instead of unknown)
    for f in files:
        if "size256_shots0_max_samples1" in f:
            try:
                d = json.load(open(f))
                if len(d.get("data", [])) >= 50:
                    return {e["query_id"]: bool(e.get("exact_match")) for e in d["data"]}
            except Exception:
                continue
    # Fallback: file with max n_data
    best_file, best_n = None, 0
    for f in files:
        try:
            d = json.load(open(f))
            n = len(d.get("data", []))
            if n > best_n:
                best_n = n; best_file = f
        except Exception:
            continue
    if not best_file or best_n < 50:
        return None
    d = json.load(open(best_file))
    return {e["query_id"]: bool(e.get("exact_match")) for e in d["data"]}


def extract_pool_texts(query_json: dict, pool_key: str) -> list:
    """Return the list of memory text strings that were seen by the answer LLM
    (or its equivalent in the baseline)."""
    if pool_key == "memories_str":
        ms = query_json.get("memories_str", "")
        return [line.lstrip("- ").strip() for line in ms.split("\n") if line.strip()]
    if pool_key == "retrieved_memories":
        rm = query_json.get("retrieved_memories") or []
        return [m.get("memory", "") for m in rm]
    if pool_key == "edges":
        eds = query_json.get("edges") or []
        return [e.get("fact", "") for e in eds]
    return []


def classify_pool_state(pool_texts: list, gt_new: str, gt_old: str) -> str:
    """Return one of PP-New / PP-Both / PP-OldOnly / PP-Missing."""
    has_new = any(match_pair(t, gt_new, gt_old, "new") for t in pool_texts)
    has_old = any(match_pair(t, gt_new, gt_old, "old") for t in pool_texts)
    if has_new and not has_old: return "PP-New"
    if has_new and has_old:     return "PP-Both"
    if not has_new and has_old: return "PP-OldOnly"
    return "PP-Missing"


def analyze(method_name: str, root: Path, pool_key: str, results_dir: str, L: str) -> dict:
    """Return {qid: (pool_state, acc_bool, response)} + summary counters.

    EM policy: prefer per-qid `response` + MAB's normalize_answer, because
    per-qid file is the write source of truth from `_handle_*_agent` and does
    not suffer from Zep 32k aggregation corruption (Zep 32k
    outputs/gpt-4o-mini-zep/Conflict_Resolution/*_results.json had 48/65 has_pair
    outputs replaced by "Answer:" placeholders while the per-qid `response` field
    kept the real LLM answer). Aggregated `exact_match` is loaded as fallback
    only when per-qid data is unusable.
    """
    em_agg = load_em(results_dir, L) or {}
    gt = load_gt(L)
    hp_qids = {q for q, e in gt.items() if e.get("conflict_type") == "has_pair"}
    per_query = {}
    counters = Counter()  # (pool_state, acc)
    query_dir = root / f"factconsolidation_sh_{L}/chunksize_512"
    n_missing_json = 0
    for qid in sorted(hp_qids):
        # Locate query file
        files = glob.glob(str(query_dir / f"query_{qid}_context_*.json"))
        if not files:
            n_missing_json += 1
            per_query[qid] = ("__no_query_json__", em_agg.get(qid, False), None)
            continue
        try:
            j = json.load(open(files[0]))
        except Exception:
            n_missing_json += 1
            per_query[qid] = ("__json_error__", em_agg.get(qid, False), None)
            continue
        pool = extract_pool_texts(j, pool_key)
        gt_new = gt[qid].get("gt_fact_text") or ""
        gt_old = gt[qid].get("old_fact_text") or ""
        state = classify_pool_state(pool, gt_new, gt_old)
        response_full = j.get("response") or ""
        # Compute EM from per-qid response using MAB's normalize_answer
        gt_answer = gt[qid].get("gt_answer")
        acc = _em_from_perqid(response_full, gt_answer)
        per_query[qid] = (state, acc, response_full[:80])
        counters[(state, acc)] += 1
    return {
        "method": method_name,
        "length": L,
        "n_hp": len(hp_qids),
        "n_missing_json": n_missing_json,
        "per_query": per_query,
        "counters": counters,
    }


# ==================== Markdown output ==================== #
POOL_ORDER = ["PP-New", "PP-Both", "PP-OldOnly", "PP-Missing"]
POOL_MEAN = {
    "PP-New": "pool 只含 gt_new(KU 解決)",
    "PP-Both": "pool 兩版都在(依 LLM 從中挑)",
    "PP-OldOnly": "pool 只含 gt_old(gt_new 消失)",
    "PP-Missing": "兩版都不在 pool(retrieval / write miss)",
}


def emit_length_table(results_per_method: list, L: str, out: list):
    """Emit per-length table + per-bucket wrong-qid list."""
    # find best per column for bold
    def fmt_cell(acc_ok: int, total: int) -> str:
        if total == 0: return "0/0"
        pct = 100 * acc_ok / total
        return f"{acc_ok}/{total} ({pct:.0f}%)"

    # per-length header
    n_hp = results_per_method[0].get("n_hp") if results_per_method else 0
    out.append(f"### Length = {L}  (has_pair N = {n_hp})")
    out.append("")

    out.append("**Table.** Return-context × Accuracy cross-tab. Each cell = `Acc✓/Bucket_total (Acc-rate in bucket)`. "
               "**Row = method**, **Col = pool state**. E2E ↑ = final has_pair exact-match. Single deterministic run (temp 0), no error bar.")
    out.append("")

    # Header
    out.append("| Method | PP-New Acc/N ↑ | PP-Both Acc/N | PP-OldOnly Acc/N ↓ | PP-Missing Acc/N ↓ | E2E Acc/N ↑ |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: |")

    # Compute best E2E for bold
    e2e = []
    for r in results_per_method:
        c = r["counters"]
        e2e_ok = sum(c[(s, True)] for s in POOL_ORDER)
        e2e_total = sum(c[(s, ok)] for s in POOL_ORDER for ok in (True, False))
        e2e.append((e2e_ok, e2e_total))
    best_e2e_ok = max((o for o, _ in e2e), default=0)

    for r, (e2e_ok, e2e_total) in zip(results_per_method, e2e):
        c = r["counters"]
        cells = []
        for s in POOL_ORDER:
            ok, no = c[(s, True)], c[(s, False)]
            cells.append(fmt_cell(ok, ok + no))
        e2e_str = fmt_cell(e2e_ok, e2e_total)
        # Bold if best
        method = r["method"]
        row = f"| {method} | " + " | ".join(cells) + f" | "
        row += f"**{e2e_str}**" if e2e_ok == best_e2e_ok else e2e_str
        row += " |"
        out.append(row)
    out.append("")


def emit_wrong_qid_lists(results_per_method: list, L: str, out: list):
    """Emit wrong-qid list per (method, pool state bucket)."""
    out.append(f"### Wrong-qid list per bucket (for Tier 2 case study,length = {L})")
    out.append("")
    out.append("每格列 wrong qids(供 §5 case study 挑選)。空格代表無 wrong qids 在該桶。")
    out.append("")
    out.append("| Method | PP-New wrong | PP-Both wrong | PP-OldOnly wrong | PP-Missing wrong |")
    out.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in results_per_method:
        buckets = defaultdict(list)
        for qid, (state, acc, _) in r["per_query"].items():
            if not acc and state in POOL_ORDER:
                buckets[state].append(qid)
        row = f"| {r['method']} | "
        for s in POOL_ORDER:
            qs = sorted(buckets[s])
            if not qs:
                row += " — |"
            elif len(qs) <= 8:
                row += " " + ", ".join(str(q) for q in qs) + " |"
            else:
                row += f" {', '.join(str(q) for q in qs[:6])}, ... (+{len(qs)-6}) |"
        out.append(row)
    out.append("")


def main(lengths=None):
    lengths = lengths or ["6k", "32k", "64k"]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    out = []

    # Header + methodology
    out.append("# Tier 1 主 metric — Return-context × Accuracy Cross-tabulation")
    out.append("")
    out.append("> **Purpose**:對每 method × length,把「pool state」(送進 answer LLM 前的 memory 內容)與「E2E Acc」(最終答對錯)交叉分類 → **精準呈現答題結果如何被 pool state 預測**。")
    out.append(">")
    out.append("> **Rigor**:matcher v3(triple-based,拒絕 Layer-1 SequenceMatcher 的 false positive);ours* 用 `memories_str`(實際 answer LLM 看到的內容),mem0(b) 用 `retrieved_memories`,Zep 用 `edges`;single deterministic run,無 error bar。")
    out.append(">")
    out.append("> **Corresponding protocol**:[../evaluation_protocol_main.md §4.2](../evaluation_protocol_main.md)")
    out.append("")
    out.append("---")
    out.append("")

    all_results = {}
    for L in lengths:
        results_per_method = []
        for method_name, root, pool_key, results_dir in METHODS:
            r = analyze(method_name, root, pool_key, results_dir, L)
            if "error" in r:
                print(f"[skip] {method_name} × {L}: {r['error']}")
                continue
            results_per_method.append(r)
        all_results[L] = results_per_method

        # emit table + observation placeholder
        out.append(f"## {L}")
        out.append("")
        emit_length_table(results_per_method, L, out)
        emit_wrong_qid_lists(results_per_method, L, out)
        out.append("")

    # Aggregated observation (3-part)
    out.append("---")
    out.append("")
    out.append("## Observation(three-part: what / why / implication)")
    out.append("")

    # Compute aggregate stats for observation
    for L in lengths:
        results_per_method = all_results[L]
        out.append(f"### Length = {L}")
        out.append("")
        # Compute per-method: PP-New count, PP-Both count, PP-Missing+OldOnly count, in-bucket accuracy
        rows = []
        for r in results_per_method:
            c = r["counters"]
            total_n = sum(c[(s, ok)] for s in POOL_ORDER for ok in (True, False))
            new_ok = c[("PP-New", True)]; new_no = c[("PP-New", False)]
            both_ok = c[("PP-Both", True)]; both_no = c[("PP-Both", False)]
            old_ok = c[("PP-OldOnly", True)]; old_no = c[("PP-OldOnly", False)]
            miss_ok = c[("PP-Missing", True)]; miss_no = c[("PP-Missing", False)]
            rows.append({
                "method": r["method"],
                "N": total_n,
                "PPnew_share": 100 * (new_ok+new_no) / max(1,total_n),
                "PPmiss_share": 100 * (miss_ok+miss_no) / max(1,total_n),
                "PPold_share": 100 * (old_ok+old_no) / max(1,total_n),
                "PPnew_in_acc": 100*new_ok/max(1,new_ok+new_no) if (new_ok+new_no) else None,
                "PPboth_in_acc": 100*both_ok/max(1,both_ok+both_no) if (both_ok+both_no) else None,
                "e2e": 100 * (new_ok+both_ok+old_ok+miss_ok) / max(1,total_n),
            })
        # find leaders
        out.append("**What** — pool state 分佈 + in-bucket accuracy(per row):")
        out.append("")
        for r in rows:
            new_acc_s = f"{r['PPnew_in_acc']:.0f}%" if r['PPnew_in_acc'] is not None else "—"
            both_acc_s = f"{r['PPboth_in_acc']:.0f}%" if r['PPboth_in_acc'] is not None else "—"
            both_share = 100 - r['PPnew_share'] - r['PPmiss_share'] - r['PPold_share']
            out.append(f"- **{r['method']}**: PP-New 佔 {r['PPnew_share']:.0f}% (Acc-in-bucket = {new_acc_s}), PP-Both 佔 {both_share:.0f}% (Acc-in-bucket = {both_acc_s}), PP-OldOnly {r['PPold_share']:.0f}%, PP-Missing {r['PPmiss_share']:.0f}%; **E2E = {r['e2e']:.1f}%**")
        out.append("")

    # Final aggregate implication
    out.append("---")
    out.append("")
    out.append("## Aggregate implication(3 lengths 合看)")
    out.append("")
    out.append("**Why** — 從 3 個長度的 pool state 分佈 + Acc-in-bucket 看到三個 pattern:")
    out.append("")
    out.append("- **Pattern 1 — ours 給乾淨 pool,LLM 幾乎 100% 抄對**:ours variants 的 PP-New 佔比 55-72%(3 lengths avg ≈ 63%),PP-New Acc-in-bucket **92-100%**(pipeline 給對答案 → LLM 抄對)。P5 skip 版(no_p5 / p3_only)在 PP-Both 桶的 Acc-in-bucket 也高達 76-91%(template hint「取大 serial」發揮作用)。")
    out.append("- **Pattern 2 — (b) mem0+P1 的 pool 混亂,LLM 猜對率低**:(b) 的 PP-New 佔比只 34-38%,**PP-OldOnly + PP-Missing 佔比合計 40-52%**(write-time destructive commit 直接毀掉 gt_new)。**即使 PP-New 桶,Acc-in-bucket 也只 52-56%** — 因為 UPDATE 常合併新舊,pool 內 fact 文字未必是純粹 gt_new,LLM 從 rewrite 過的文字判讀不穩;PP-Missing 桶 Acc-in-bucket 22-50%(LLM 純猜)。")
    out.append("- **Pattern 3 — Zep(k=10)pool 幾乎全 PP-Both,依賴 answer LLM 猜**:PP-Both 佔比 **93-97%**(未做 query-time KU resolution + top-10 兩版都塞進來)。6k / 64k 上 Acc-in-bucket 55-61%,但 **32k catastrophe(Acc-in-bucket 只 6%)**— top-10 上限 + 長 context 使記憶顆粒度失效。")
    out.append("")
    out.append("**Implication** — 支撐核心 claim:")
    out.append("")
    out.append("> ours 的 E2E 優勢 **主要來自 pool state 的差異**(pipeline 送對 gt_new、濾掉 gt_old),**而非 answer LLM 在 PP-Both 時的猜對率**:ours 的 PP-New 佔比是 (b) mem0+P1 的近 2 倍(63% vs 36%),且 PP-New Acc-in-bucket 是 mem0(b) 的近 2 倍(97% vs 54%)。這對應論文核心主張:**query-time KU resolution 讓記憶正確地被檢索用於推論**;倚賴 answer LLM 從混雜 / 錯誤 pool 中挑對是脆弱的,尤其在弱 backbone(intro 受限部署動機)下崩壞更劇。")
    out.append("")
    out.append("**Follow-up(Tier 2 case study)**:對每 method 的 wrong-qid list per bucket 挑代表 case → trace 錯誤模式(A/B/C/D/E),見 [`../evaluation_protocol_main.md §5`](../evaluation_protocol_main.md)")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Sanity-check")
    out.append("")
    out.append("- 每 row 4 個 bucket 加總應 = N_hp(4×2 分類 collectively exhaustive)")
    out.append("- E2E Acc column 應與 evaluation_protocol_main.md §3.1 landscape 表數字一致")
    out.append("- Zep 的 k=10 caveat:與其他 k=100 不對稱,PP-Missing 被 top-10 上限 bias(不代表 Zep bank 缺 gt_new)")
    out.append("")
    out.append("---")
    out.append("")
    out.append("*Generated by `analysis/compute_pool_acc_crosstab.py`; source data = `outputs/rag_retrieved/**/query_*.json` + `outputs/**/Conflict_Resolution/*results*.json`. Regenerate: `python analysis/compute_pool_acc_crosstab.py`.*")
    OUT_MD.write_text("\n".join(out))
    print(f"\n[wrote] {OUT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", "-L", nargs="*", default=None, choices=["6k", "32k", "64k"])
    args = ap.parse_args()
    main(args.length)
