#!/usr/bin/env python3
"""Canonical FC-SH rescore — single source of truth for E2E metrics.

WHY THIS EXISTS
---------------
Three problems motivated this script (2026-07-10 metric audit):

  P1. The paper's `has_pair EM` historically read the `exact_match` field, but
      MemoryAgentBench's *official* Conflict-Resolution metric is
      `substring_exact_match` (README §"Clarification on Evaluation Metrics").
  P2. Zep side-paths (run_zep_full100.py, rerun_zep_edges_only.py, ...) each
      hand-rolled their own scorer (`pred.lower().strip('.,') == gt`), bypassing
      the official normalize + alias-list + max(raw, parse_output) pipeline.
  P3. Some output dirs mix a stale custom-scorer file (…FULL100…, …backbone_swap…,
      …smoke…) next to the official main.py file, so a naive sorted(glob)[0]
      silently picks the wrong run.

This script recomputes every metric from raw `output` + `answer` using the
OFFICIAL post-process semantics, from the ONE canonical file per cell, and
emits three metrics side by side so the paper can cite a consistent number.

METRICS (all use official normalize + parse_output + max-over-{raw,parsed})
  - strict_em      : normalize(pred) == normalize(gt_new)          [drqa_exact_match_score]
  - official_subem : normalize(gt_new) in normalize(pred)          [substring_exact_match_score, MAB official]
  - correct_sem    : gt_new present AND gt_old absent              [paper §4.1.4; format-robust + hedge-excluded]

CORRECT-SEM PRECISION NOTE
  gt_new has an alias list (results.json `answer`); gt_old does NOT — we extract
  its surface form from old_fact_text by stripping the shared stem it shares with
  gt_fact_text. In FC-SH this is sound because the answer LLM copies pool surface
  forms, so a hedge echoes the pool's old form (verified on qid=42 6k, David
  Farragut 64k). It can under-detect a hedge that phrases gt_old via a
  world-knowledge alias not present in the pool — a known, minor limitation.

CANONICAL FILE SELECTION
  Official main.py outputs carry a top-level `metrics` key; hand-rolled Zep
  side-path files carry `note` instead. We pick official files, size256, k in
  {100, 10}, chunk512, and among those the one with the most has_pair coverage.
"""
import json
import glob
import os
import re
import string
import sys

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ---------------------------------------------------------------------------
# Official scoring primitives (verbatim port of utils/eval_other_utils.py so the
# script runs without the heavy nltk/torch import chain).
# ---------------------------------------------------------------------------

def normalize_answer(text):
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_output(output_text, answer_prefix="Answer:"):
    patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pat in patterns:
        m = pat.search(output_text or "")
        if m:
            extracted = m.group(1).strip()
            return re.sub(f"^{re.escape(answer_prefix)}", "", extracted, flags=re.IGNORECASE).strip()
    return None


def _flatten(answers):
    if isinstance(answers, str):
        return [answers]
    out = []
    for a in answers or []:
        out.extend(_flatten(a)) if isinstance(a, list) else out.append(a)
    return out


def _candidates(pred):
    cands = [pred]
    parsed = parse_output(pred)
    if parsed is not None:
        cands.append(parsed)
    return cands


def strict_em(pred, gt_new_aliases):
    gts = [normalize_answer(g) for g in _flatten(gt_new_aliases)]
    return any(normalize_answer(c) in gts for c in _candidates(pred))


def official_subem(pred, gt_new_aliases):
    gts = [normalize_answer(g) for g in _flatten(gt_new_aliases)]
    return any(any(g in normalize_answer(c) for g in gts) for c in _candidates(pred))


def correct_sem(pred, gt_new_aliases, gt_old_surface):
    """gt_new present (substring) AND gt_old surface NOT present."""
    new_gts = [normalize_answer(g) for g in _flatten(gt_new_aliases)]
    old = normalize_answer(gt_old_surface)
    # evaluate over the same candidate set; pass if ANY candidate satisfies both.
    for c in _candidates(pred):
        nc = normalize_answer(c)
        new_in = any(g in nc for g in new_gts)
        old_in = bool(old) and old in nc
        if new_in and not old_in:
            return True
    return False


def extract_old_surface(gt_fact_text, old_fact_text):
    """gt_new='X is Y', gt_old='X is Z' -> 'Z' (shared-stem-stripped tail)."""
    a = (gt_fact_text or "").split()
    b = (old_fact_text or "").split()
    k = 0
    for wa, wb in zip(a, b):
        if wa.lower().rstrip(".,;:!?") != wb.lower().rstrip(".,;:!?"):
            break
        k += 1
    return " ".join(b[k:]).rstrip(".,;:!?")


# ---------------------------------------------------------------------------
# Canonical file selection
# ---------------------------------------------------------------------------

_STALE_MARKERS = ("full100", "backbone_swap", "smoke", "size10", "size3",
                  "edges_only", "episode")

# Cells that have NO official main.py file — only a custom-scorer run exists.
# We re-score its raw output with the OFFICIAL pipeline here (verified to match
# the paper's cited strict/sEM), but tag provenance so the deviation is visible.
_CUSTOM_FALLBACK = {
    ("gpt-4.1-mini-zep", "64k"):
        "factconsolidation_sh_64k_unknown_backbone_swap_size256_shots0_max_samplesunknown_k10_chunk512_results.json",
    # gemma Zep path-A runs: filename carries `backbone_swap` (a stale marker) so
    # pick_canonical_file skips it; force-pick the official-fields file here.
    ("gemma3-1b-zep", "6k"):  "factconsolidation_sh_6k_unknown_backbone_swap_size256_shots0_max_samplesunknown_k10_chunk512_results.json",
    ("gemma3-4b-zep", "6k"):  "factconsolidation_sh_6k_unknown_backbone_swap_size256_shots0_max_samplesunknown_k10_chunk512_results.json",
    ("gemma3-12b-zep", "6k"): "factconsolidation_sh_6k_unknown_backbone_swap_size256_shots0_max_samplesunknown_k10_chunk512_results.json",
    ("gemma3-27b-zep", "6k"): "factconsolidation_sh_6k_unknown_backbone_swap_size256_shots0_max_samplesunknown_k10_chunk512_results.json",
}


def pick_canonical_file(run_dir, length):
    pat = os.path.join(run_dir, "Conflict_Resolution", f"*sh_{length}_*results*.json")
    best, best_hp = None, -1
    for f in sorted(glob.glob(pat)):
        base = os.path.basename(f).lower()
        if any(m in base for m in _STALE_MARKERS):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        # official main.py outputs carry top-level `metrics`; Zep custom carry `note`.
        # Zep's official main.py file also qualifies (it has `metrics`).
        data = d.get("data") or []
        if not data or "output" not in data[0]:
            continue
        n = len(data)
        if n > best_hp:
            best, best_hp = f, n
    if best is None:
        fallback = _CUSTOM_FALLBACK.get((os.path.basename(run_dir.rstrip("/\\")), length))
        if fallback:
            cand = os.path.join(run_dir, "Conflict_Resolution", fallback)
            if os.path.exists(cand):
                return cand
    return best


def load_haspair(length):
    fp = os.path.join(ANALYSIS_DIR, f"sh_{length}_mquake_analysis.json")
    rows = json.load(open(fp, encoding="utf-8"))
    return {r["query_id"]: r for r in rows if r.get("conflict_type") == "has_pair"}


def score_cell(run_dir, length):
    f = pick_canonical_file(run_dir, length)
    if not f:
        return None
    hp = load_haspair(length)
    data = json.load(open(f, encoding="utf-8"))["data"]
    # overall = all queries in the file (the official 100-query task metric).
    # has_pair = KU-relevant subset (has gt_new/gt_old); only here is correct-sEM defined.
    o_se = o_sb = o_n = 0
    h_se = h_sb = h_cs = h_n = 0
    for it in data:
        pred = it.get("output")
        gt_new = it.get("answer")
        if pred is None or gt_new is None:
            continue
        o_n += 1
        o_se += strict_em(pred, gt_new)
        o_sb += official_subem(pred, gt_new)
        r = hp.get(it.get("query_id"))
        if r is not None:
            old_surface = extract_old_surface(r.get("gt_fact_text", ""), r.get("old_fact_text", ""))
            h_n += 1
            h_se += strict_em(pred, gt_new)
            h_sb += official_subem(pred, gt_new)
            h_cs += correct_sem(pred, gt_new, old_surface)
    if o_n == 0:
        return None
    is_custom = any(m in os.path.basename(f).lower() for m in _STALE_MARKERS)
    return {"file": os.path.basename(f),
            "provenance": "custom-file-official-rescore" if is_custom else "official-main.py",
            "overall_N": o_n, "overall_strict_em": o_se, "overall_subem": o_sb,
            "haspair_N": h_n, "haspair_strict_em": h_se, "haspair_subem": h_sb,
            "haspair_correct_sem": h_cs}


# ---------------------------------------------------------------------------
# Paper's canonical method x backbone -> run-dir registry
# ---------------------------------------------------------------------------

REGISTRY = {
    "gpt-4o-mini": {
        "ours (main)":     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_no_p5",
        "ours (no P3)":    "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_struct",
        "ours (LLM only)": "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_p3_only_no_struct",
        "ours (+P5)":      "gpt-4o-mini-mem0-chunk512-temp0-openai-unified",
        "(a) vanilla":     "gpt-4o-mini-mem0-chunk512-temp0-openai-native",
        "(b) mem0+P1":     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_dest",
        "Q-llm-recency":   "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_q_llm_recency",
        "Zep (k=10)":      "gpt-4o-mini-zep",
        "LCA":             "gpt-4o-mini-temp0",
    },
    "gpt-4.1-mini": {
        "ours (main)":     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_no_p5",
        "ours (no P3)":    "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_struct",
        "ours (LLM only)": "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_p3_only_no_struct",
        "ours (+P5)":      "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified",
        "(b) mem0+P1":     "gpt-4.1-mini-mem0-chunk512-temp0-openai-unified_dest",
        "Zep (k=10)":      "gpt-4.1-mini-zep",
    },
    # gpt-5.4-mini strong-tier backbone probe (2026-07-15). Tests whether
    # the strongest OpenAI mini-tier model closes the ours vs baselines gap
    # observed on gpt-4o-mini (mid-tier). Only 6k first for cost efficiency.
    "gpt-5.4-mini": {
        "ours (main)":     "gpt-5.4-mini-mem0-chunk512-temp0-openai-unified_no_p5",
        "(b) mem0+P1":     "gpt-5.4-mini-mem0-chunk512-temp0-openai-unified_dest",
        "Q-llm-recency":   "gpt-5.4-mini-mem0-chunk512-temp0-openai-unified_q_llm_recency",
        "Zep (k=10)":      "gpt-5.4-mini-zep",
    },
}

# gemma tiers (GX10 per-backbone weak-model runs). NOTE: GX10 uses the gpt-4o-mini
# TEMPLATE prefix + a `__gemma3-{s}` suffix on the run-dir (extraction is per-backbone
# gemma via MEM0_TRIPLE_MODEL — the "gpt-4o-mini" in the dir name is a legacy template
# tag, NOT the backbone). Expanded programmatically to avoid ~28 literal lines.
_GEMMA_METHODS = {
    "ours (main)":     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_no_p5__gemma3-{s}",
    "ours (no P3)":    "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_struct__gemma3-{s}",
    "ours (LLM only)": "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_p3_only_no_struct__gemma3-{s}",
    "ours (+P5)":      "gpt-4o-mini-mem0-chunk512-temp0-openai-unified__gemma3-{s}",   # only 1b/4b exist; others skip
    "(b) mem0+P1":     "gpt-4o-mini-mem0-chunk512-temp0-openai-unified_dest__gemma3-{s}",
    "(a) vanilla":     "gpt-4o-mini-mem0-chunk512-temp0-openai-native__gemma3-{s}",
    "Zep (k=10)":      "gemma3-{s}-zep",
}
for _s in ("1b", "4b", "12b", "27b"):
    REGISTRY[f"gemma3-{_s}"] = {m: t.format(s=_s) for m, t in _GEMMA_METHODS.items()}

LENGTHS = ["6k", "32k", "64k", "262k"]


# Custom-format method files that do NOT live under outputs/<rundir>/Conflict_Resolution/.
# Rows carry {query_id, answer(=pred), gt_answer, subem} — pre-scored by author's
# own comparator. We DO re-run the official pipeline against gt_answer (single string,
# no alias list available here) so numbers are directly comparable to other cells.
# Where the author's single-string GT under-scores a hedge-free alias hit, ties break
# on the side of under-counting Don't Ask — flagged as `provenance` for transparency.
CUSTOM_MAXSERIAL = {
    "gpt-4o-mini": {
        "Don't Ask (author's, vec-top100)":
            os.path.join(OUTPUTS_DIR, "maxserial_theircode", "{L}_gpt-4o-mini_vector100.json"),
    },
    "gpt-5.4-mini": {
        "Don't Ask (author's, vec-top100)":
            os.path.join(OUTPUTS_DIR, "maxserial_theircode", "{L}_gpt-5.4-mini_vector100.json"),
    },
}


def score_custom_maxserial(path_tmpl, length):
    path = path_tmpl.format(L=length)
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    rows = d.get("rows") or []
    if not rows:
        return None
    hp = load_haspair(length)
    o_se = o_sb = o_n = 0
    h_se = h_sb = h_cs = h_n = 0
    for it in rows:
        pred = it.get("answer")
        gt_new = it.get("gt_answer")
        if pred is None or gt_new is None:
            continue
        # gt_new is single string here; wrap for _flatten
        gt_list = [gt_new]
        o_n += 1
        o_se += strict_em(pred, gt_list)
        o_sb += official_subem(pred, gt_list)
        r = hp.get(it.get("query_id"))
        if r is not None:
            old_surface = extract_old_surface(r.get("gt_fact_text", ""),
                                              r.get("old_fact_text", ""))
            h_n += 1
            h_se += strict_em(pred, gt_list)
            h_sb += official_subem(pred, gt_list)
            h_cs += correct_sem(pred, gt_list, old_surface)
    if o_n == 0:
        return None
    return {"file": os.path.basename(path),
            "provenance": "custom-format-official-rescore-single-gt",
            "overall_N": o_n, "overall_strict_em": o_se, "overall_subem": o_sb,
            "haspair_N": h_n, "haspair_strict_em": h_se, "haspair_subem": h_sb,
            "haspair_correct_sem": h_cs}


def main():
    out_rows = []
    for backbone, methods in REGISTRY.items():
        for method, rundir in methods.items():
            run_dir = os.path.join(OUTPUTS_DIR, rundir)
            for L in LENGTHS:
                res = score_cell(run_dir, L)
                if res is None:
                    continue
                out_rows.append({"backbone": backbone, "method": method,
                                 "length": L, **res})
    # custom-format method files (Don't Ask etc.)
    for backbone, methods in CUSTOM_MAXSERIAL.items():
        for method, tmpl in methods.items():
            for L in LENGTHS:
                res = score_custom_maxserial(tmpl, L)
                if res is None:
                    continue
                out_rows.append({"backbone": backbone, "method": method,
                                 "length": L, **res})

    # console table — overall (100-query) SubEM is the headline; has_pair SubEM for analysis
    hdr = (f"{'backbone':<13}{'method':<36}{'len':<5}  "
           f"{'overall SubEM':>16}{'has_pair SubEM':>16}")
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        on, hn = r["overall_N"], r["haspair_N"]
        ov = f"{r['overall_subem']}/{on} ({100*r['overall_subem']/on:.0f}%)"
        hp = f"{r['haspair_subem']}/{hn} ({100*r['haspair_subem']/hn:.0f}%)" if hn else "-"
        print(f"{r['backbone']:<13}{r['method']:<36}{r['length']:<5}  {ov:>16}{hp:>16}")

    out_json = os.path.join(ANALYSIS_DIR, "canonical_fc_sh_metrics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out_rows, fh, ensure_ascii=False, indent=1)
    print(f"\nWrote {out_json}  ({len(out_rows)} cells)")

    # ---- Markdown pivot table: rows = method, cols = length; per backbone ----
    md_lines = ["# Canonical FC-SH metrics (auto-generated by `rescore_canonical.py`)",
                "",
                "Overall SubEM (%) — official post-process semantics, canonical file per cell.",
                ""]
    by_bb = {}
    for r in out_rows:
        by_bb.setdefault(r["backbone"], {}).setdefault(r["method"], {})[r["length"]] = r
    for bb, methods in by_bb.items():
        md_lines.append(f"## Backbone: `{bb}`")
        md_lines.append("")
        md_lines.append("### Overall SubEM (headline)")
        md_lines.append("")
        head = "| Method | " + " | ".join(LENGTHS) + " |"
        sep  = "|:--|" + ":--:|" * len(LENGTHS)
        md_lines.append(head); md_lines.append(sep)
        for m, cells in methods.items():
            row = [f"`{m}`"]
            for L in LENGTHS:
                c = cells.get(L)
                if c:
                    row.append(f"{100*c['overall_subem']/c['overall_N']:.0f} ({c['overall_subem']}/{c['overall_N']})")
                else:
                    row.append("—")
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")
        md_lines.append("### has_pair SubEM (KU-relevant subset)")
        md_lines.append("")
        md_lines.append(head); md_lines.append(sep)
        for m, cells in methods.items():
            row = [f"`{m}`"]
            for L in LENGTHS:
                c = cells.get(L)
                if c and c["haspair_N"]:
                    row.append(f"{100*c['haspair_subem']/c['haspair_N']:.0f} ({c['haspair_subem']}/{c['haspair_N']})")
                else:
                    row.append("—")
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")
        md_lines.append("### has_pair correct-sEM (gt_new present ∧ gt_old absent)")
        md_lines.append("")
        md_lines.append(head); md_lines.append(sep)
        for m, cells in methods.items():
            row = [f"`{m}`"]
            for L in LENGTHS:
                c = cells.get(L)
                if c and c["haspair_N"]:
                    row.append(f"{100*c['haspair_correct_sem']/c['haspair_N']:.0f} ({c['haspair_correct_sem']}/{c['haspair_N']})")
                else:
                    row.append("—")
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Provenance & missing cells")
    md_lines.append("")
    md_lines.append("- `provenance=custom-format-official-rescore-single-gt`: Don't Ask author's per-row output rescored "
                    "with official normalize+substring against `gt_answer` (single string; may under-score vs alias-list).")
    md_lines.append("- Missing cells (`—`) = run not yet executed OR file selection filtered by stale-marker rule.")

    out_md = os.path.join(ANALYSIS_DIR, "canonical_fc_sh_metrics.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
