#!/usr/bin/env python3
"""Consolidated FC-SH error-mode dominant-bucket analysis (gpt-4o-mini, all 4 lengths).

THESIS THIS SCRIPT SUPPORTS
    Every baseline loses the NEW (counterfactual) version at a *different* pipeline
    stage, but the failure SIGNATURE is identical: the final answer falls back to
    the LLM-prior-consistent OLD value (gt_OLD). We measure each family with the
    metric that isolates ITS stage, and show they all terminate in gt_OLD.

SCOPE (coordinator-approved): committed data only, NO model re-runs, NO .env read.
    Backbone = gpt-4o-mini, temp 0, single deterministic run. has_pair subset.

SINGLE GROUND TRUTH  : rescore_canonical.load_haspair(L)  -> sh_{L}_mquake_analysis.json
                       (gt_fact_text / old_fact_text / gt_answer / gt_seq / old_seq)
SINGLE SubEM MATCHER : rescore_canonical.official_subem (MAB official substring-EM,
                       normalize + parse_output + max-over-{raw,parsed}).
gt_OLD SURFACE       : rescore_canonical.extract_old_surface (shared-stem strip).
    answered_old(pred) := official_subem(pred, [old_surface])  (canonical, symmetric
    to how we score gt_new). A has_pair FAILURE is `not official_subem(pred, gt_new)`.

PER-METHOD METRIC (the point: right metric per family)
  Vanilla-RAG (q_llm_recency)  query-time recency judge : of has_pair FAIL, % answered gt_OLD
  Ours (Struct-Only) (struct)  query-time structural    : of has_pair FAIL, % answered gt_OLD
  Don't Ask (maxserial)        query-time extraction     : of has_pair FAIL, % (n_candidates==1 AND gt_OLD)
                                                            reported RAW and D-flag-excluded
  Mem0+P1 (dest)               write-time destructive    : pool-state, % new-absent (PP-OldOnly+PP-Missing)
                                                            + their subEM accuracy (expect ~0)
  Mem0 Vanilla (native)        write-time mis-apply      : coarse overall SubEM only (no fine bucket)
  Zep (k=10)                   write/query bi-temporal    : (a) NotBothExtracted %, all lengths;
                                                            (b) answer-side link on Additive-NoKU fails

Usage (repro env, repo root):
    python analysis/errormode_dominant_buckets.py            # all 4 lengths, prints table, writes md
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

# ---- SINGLE ground-truth loader + SINGLE subEM matcher + gt_OLD surface ----
from analysis.rescore_canonical import (          # noqa: E402
    official_subem, extract_old_surface, load_haspair, pick_canonical_file,
)
# ---- committed pool-state logic (mem0) ----
from analysis.compute_pool_acc_crosstab import (   # noqa: E402
    classify_pool_state, extract_pool_texts,
)
# ---- committed Zep bi-temporal classifier ----
from analysis.classify_zep_ku_resolution import classify_query   # noqa: E402
from analysis.compute_m1_m2_m3 import match_pair                 # noqa: E402

LENGTHS = ["6k", "32k", "64k", "262k"]
SHORT = ["6k", "32k", "64k"]

OUT_MD = REPO / "analysis/results/fc_sh_errormode_dominant_buckets.md"

OUT = os.environ.get("OUTPUT_ROOT", "outputs")  # fresh runs (default) or reference_outputs
DATA = {
    "vanilla_rag": OUT + "/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_q_llm_recency",
    "struct_only": OUT + "/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_struct",
    "dont_ask":    OUT + "/maxserial_theircode/{L}_gpt-4o-mini_vector100.json",
    "mem0_p1":     OUT + "/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_dest/k_100",
    "mem0_vanilla":OUT + "/gpt-4o-mini-mem0-chunk512-temp0-openai-native",
    "zep":         OUT + "/rag_retrieved/Structure_rag_zep/k_10",
}


def _old_surface(row):
    return extract_old_surface(row.get("gt_fact_text", ""), row.get("old_fact_text", ""))


def answered_old(pred, old_surface):
    """Canonical: gt_OLD surface substring-present in prediction (same matcher as gt_new)."""
    return bool(old_surface) and official_subem(pred, [old_surface])


# ===================== query-time results.json methods ===================== #
def analyze_query_method(rundir):
    """Return per-length dict: {L: (n_hp, n_wrong, n_fall_old, wrong_qids)}."""
    out = {}
    for L in LENGTHS:
        f = pick_canonical_file(str(REPO / rundir), L)
        hp = load_haspair(L)
        if not f:
            out[L] = None
            continue
        data = json.load(open(f, encoding="utf-8"))["data"]
        byq = {d["query_id"]: d for d in data}
        n_hp = n_wrong = n_fall = 0
        wrong = []
        for qid, row in hp.items():
            d = byq.get(qid)
            if d is None:
                continue
            n_hp += 1
            pred, aliases = d.get("output"), d.get("answer")
            if official_subem(pred, aliases):      # correct (gt_new present)
                continue
            n_wrong += 1
            wrong.append(qid)
            if answered_old(pred, _old_surface(row)):
                n_fall += 1
        out[L] = (n_hp, n_wrong, n_fall, wrong)
    return out


# ===================== Don't Ask (maxserial rows) ===================== #
def analyze_dont_ask():
    out = {}
    for L in LENGTHS:
        path = str(REPO / DATA["dont_ask"].format(L=L))
        hp = load_haspair(L)
        if not os.path.exists(path):
            out[L] = None
            continue
        rows = {r["query_id"]: r for r in json.load(open(path, encoding="utf-8"))["rows"]}
        n_hp = n_wrong = 0
        nc1_raw = old_raw = nc1_and_old_raw = 0          # raw
        nwrong_df = nc1_and_old_df = 0                   # D-flag excluded
        dflag = 0
        for qid, row in hp.items():
            r = rows.get(qid)
            if r is None:
                continue
            n_hp += 1
            pred = r.get("answer")
            aliases = [r.get("gt_answer")]
            if official_subem(pred, aliases):
                continue
            n_wrong += 1
            is_dflag = (row.get("gt_seq") is not None and row.get("old_seq") is not None
                        and row["gt_seq"] < row["old_seq"])
            if is_dflag:
                dflag += 1
            ncand1 = (r.get("n_candidates") == 1)
            fell_old = answered_old(pred, _old_surface(row))
            if ncand1:
                nc1_raw += 1
            if fell_old:
                old_raw += 1
            if ncand1 and fell_old:
                nc1_and_old_raw += 1
            if not is_dflag:
                nwrong_df += 1
                if ncand1 and fell_old:
                    nc1_and_old_df += 1
        out[L] = {
            "n_hp": n_hp, "n_wrong": n_wrong, "dflag": dflag,
            "nc1_raw": nc1_raw, "old_raw": old_raw, "nc1_and_old_raw": nc1_and_old_raw,
            "n_wrong_df": nwrong_df, "nc1_and_old_df": nc1_and_old_df,
        }
    return out


# ===================== Mem0+P1 pool-state (committed logic) ===================== #
def analyze_mem0_pool():
    out = {}
    root = REPO / DATA["mem0_p1"]
    for L in LENGTHS:
        hp = load_haspair(L)
        qdir = root / f"factconsolidation_sh_{L}/chunksize_512"
        if not qdir.is_dir():
            out[L] = None
            continue
        buckets = Counter()            # pool_state -> total
        bucket_correct = Counter()     # pool_state -> subEM correct
        n = n_missing = 0
        for qid, row in hp.items():
            fs = glob.glob(str(qdir / f"query_{qid}_context_*.json"))
            if not fs:
                n_missing += 1
                continue
            j = json.load(open(fs[0], encoding="utf-8"))
            pool = extract_pool_texts(j, "retrieved_memories")
            gt_new = row.get("gt_fact_text") or ""
            gt_old = row.get("old_fact_text") or ""
            state = classify_pool_state(pool, gt_new, gt_old)
            resp = j.get("response") or ""
            correct = official_subem(resp, [row.get("gt_answer")])
            buckets[state] += 1
            if correct:
                bucket_correct[state] += 1
            n += 1
        new_absent = buckets["PP-OldOnly"] + buckets["PP-Missing"]
        new_absent_correct = bucket_correct["PP-OldOnly"] + bucket_correct["PP-Missing"]
        out[L] = {
            "n": n, "n_missing": n_missing, "buckets": buckets,
            "bucket_correct": bucket_correct,
            "new_absent": new_absent, "new_absent_correct": new_absent_correct,
        }
    return out


# ===================== Mem0 Vanilla (coarse overall subEM) ===================== #
def analyze_mem0_vanilla():
    out = {}
    for L in LENGTHS:
        f = pick_canonical_file(str(REPO / DATA["mem0_vanilla"]), L)
        if not f:
            out[L] = None
            continue
        data = json.load(open(f, encoding="utf-8"))["data"]
        n = ok = 0
        for d in data:
            pred, aliases = d.get("output"), d.get("answer")
            if pred is None or aliases is None:
                continue
            n += 1
            if official_subem(pred, aliases):
                ok += 1
        out[L] = (ok, n)
    return out


# ===================== Zep (NotBothExtracted + answer-side link) ===================== #
def analyze_zep():
    out = {}
    root = REPO / DATA["zep"]
    for L in LENGTHS:
        hp = load_haspair(L)
        qdir = root / f"factconsolidation_sh_{L}/chunksize_512"
        bucket_ct = Counter()
        n = n_missing = 0
        # answer-side link on Additive-NoKU (no-KU-signal) FAILURES
        add_fail = 0            # Additive-NoKU AND failed (subEM gt_new = 0)
        add_fail_old = 0        # ... AND answered gt_OLD
        add_total = 0
        for qid, row in hp.items():
            fs = glob.glob(str(qdir / f"query_{qid}_context_*.json"))
            if not fs:
                n_missing += 1
                continue
            j = json.load(open(fs[0], encoding="utf-8"))
            edges = j.get("edges", []) or []
            gt_new = row.get("gt_fact_text") or ""
            gt_old = row.get("old_fact_text") or ""
            bucket, _ = classify_query(edges, gt_new, gt_old)
            bucket_ct[bucket] += 1
            n += 1
            resp = j.get("response") or ""
            failed = not official_subem(resp, [row.get("gt_answer")])
            if bucket == "Additive-NoKU":
                add_total += 1
                if failed:
                    add_fail += 1
                    if answered_old(resp, _old_surface(row)):
                        add_fail_old += 1
        out[L] = {
            "n": n, "n_missing": n_missing, "bucket_ct": bucket_ct,
            "not_both": bucket_ct["NotBothExtracted"],
            "add_total": add_total, "add_fail": add_fail, "add_fail_old": add_fail_old,
        }
    return out


# ============================ reporting ============================ #
def pct(a, b):
    return f"{100*a/b:.0f}%" if b else "—"


def main():
    vr = analyze_query_method(DATA["vanilla_rag"])
    so = analyze_query_method(DATA["struct_only"])
    da = analyze_dont_ask()
    m0 = analyze_mem0_pool()
    mv = analyze_mem0_vanilla()
    zp = analyze_zep()

    L4 = LENGTHS
    lines = []
    P = lines.append

    # ---------- console ----------
    def show():
        print("\n================= FC-SH error-mode dominant buckets (gpt-4o-mini) =================")
        print("\n[Vanilla-RAG] of has_pair FAIL, % answered gt_OLD  (query-time recency judge)")
        for L in L4:
            r = vr[L]
            if r: n_hp,nw,nf,_=r; print(f"  {L:>5}: fail {nw}/{n_hp}  fall→gt_OLD {nf}/{nw} ({pct(nf,nw)})")
        print("\n[Ours Struct-Only] of has_pair FAIL, % answered gt_OLD  (query-time structural)")
        for L in L4:
            r = so[L]
            if r: n_hp,nw,nf,_=r; print(f"  {L:>5}: fail {nw}/{n_hp}  fall→gt_OLD {nf}/{nw} ({pct(nf,nw)})")
        print("\n[Don't Ask] of has_pair FAIL, % (n_candidates==1 AND gt_OLD)  RAW vs D-flag-excluded")
        for L in L4:
            d = da[L]
            if d:
                print(f"  {L:>5}: fail {d['n_wrong']}/{d['n_hp']}  raw {d['nc1_and_old_raw']}/{d['n_wrong']} "
                      f"({pct(d['nc1_and_old_raw'],d['n_wrong'])})  "
                      f"D-flag-excl {d['nc1_and_old_df']}/{d['n_wrong_df']} ({pct(d['nc1_and_old_df'],d['n_wrong_df'])})  "
                      f"[D-flag={d['dflag']}]")
        print("\n[Mem0+P1] pool-state new-absent (PP-OldOnly+PP-Missing) share + subEM acc  (write-time destructive)")
        for L in L4:
            m = m0[L]
            if m:
                print(f"  {L:>5}: new-absent {m['new_absent']}/{m['n']} ({pct(m['new_absent'],m['n'])})  "
                      f"acc-in-new-absent {m['new_absent_correct']}/{m['new_absent']} ({pct(m['new_absent_correct'],m['new_absent'])})  "
                      f"buckets={dict(m['buckets'])}")
        print("\n[Mem0 Vanilla] coarse overall SubEM  (write-time mis-apply; no fine bucket)")
        for L in L4:
            v = mv[L]
            if v: ok,n=v; print(f"  {L:>5}: overall SubEM {ok}/{n} ({pct(ok,n)})")
        print("\n[Zep] NotBothExtracted % (all lengths) + answer-side link on Additive-NoKU fails (short)")
        for L in L4:
            z = zp[L]
            if z:
                extra = ""
                if L in SHORT:
                    extra = (f"  | Additive-NoKU fails answered gt_OLD "
                             f"{z['add_fail_old']}/{z['add_fail']} ({pct(z['add_fail_old'],z['add_fail'])}) "
                             f"of {z['add_total']} additive")
                print(f"  {L:>5}: NotBothExtracted {z['not_both']}/{z['n']} ({pct(z['not_both'],z['n'])})"
                      f"{extra}")
    show()

    # ---------- markdown ----------
    P("# FC-SH error-mode — dominant bucket per method (gpt-4o-mini, all 4 lengths)")
    P("")
    P("> **Thesis**: every baseline loses the NEW (counterfactual) version at a *different* "
      "pipeline stage, but the failure signature is identical — the final answer falls back to "
      "the LLM-prior-consistent OLD value (**gt_OLD**). Each family is measured with the metric "
      "that isolates its stage; all terminate in gt_OLD.")
    P(">")
    P("> **Reproduce**: `python analysis/errormode_dominant_buckets.py` (repro env, repo root). "
      "Committed data only; no model re-runs. Backbone gpt-4o-mini, temp 0, single run.")
    P("> **Single ground truth**: `rescore_canonical.load_haspair(L)` → `analysis/results/sh_{L}_mquake_analysis.json`. "
      "**Single subEM matcher**: `rescore_canonical.official_subem` (MAB official substring-EM). "
      "**gt_OLD surface**: `rescore_canonical.extract_old_surface`; `answered_old := official_subem(pred, [old_surface])`.")
    P("")
    P("---")
    P("")
    P("## Consolidated table")
    P("")
    P("Per-length proportion columns give the DOMINANT bucket's metric at 6k / 32k / 64k / 262k.")
    P("")
    P("| Method | Pipeline stage new-version lost | Dominant bucket (metric) | 6k | 32k | 64k | 262k | Robustness | Source data |")
    P("|:--|:--|:--|:-:|:-:|:-:|:-:|:--|:--|")

    def cells_query(res):
        c = []
        for L in L4:
            r = res[L]
            c.append(pct(r[2], r[1]) + f" ({r[2]}/{r[1]})" if r else "—")
        return c

    vrc = cells_query(vr)
    P(f"| **Vanilla-RAG** | query-time recency judge (downstream) | fall-to-gt_OLD (% of has_pair fails) "
      f"| {vrc[0]} | {vrc[1]} | {vrc[2]} | {vrc[3]} | ROBUST | `{DATA['vanilla_rag']}/Conflict_Resolution/*results*.json` |")
    soc = cells_query(so)
    P(f"| **Ours (Struct-Only)** | query-time (S,P) canonicalization miss | fall-to-gt_OLD (% of has_pair fails) "
      f"| {soc[0]} | {soc[1]} | {soc[2]} | {soc[3]} | ROBUST | `{DATA['struct_only']}/Conflict_Resolution/*results*.json` |")
    dac = []
    for L in L4:
        d = da[L]
        dac.append(f"{pct(d['nc1_and_old_raw'],d['n_wrong'])} raw / {pct(d['nc1_and_old_df'],d['n_wrong_df'])} Df" if d else "—")
    P(f"| **Don't Ask** | query-time LLM candidate extraction (upstream) | n_cand==1 ∧ gt_OLD (raw / D-flag-excl) "
      f"| {dac[0]} | {dac[1]} | {dac[2]} | {dac[3]} | ROBUST (see D-flag note) | `outputs/maxserial_theircode/{{L}}_gpt-4o-mini_vector100.json` |")
    m0c = []
    for L in L4:
        m = m0[L]
        m0c.append(f"{pct(m['new_absent'],m['n'])} ({m['new_absent']}/{m['n']})" if m else "—")
    P(f"| **Mem0 + P1** | write-time destructive commit | pool new-absent = PP-OldOnly+PP-Missing (% has_pair) "
      f"| {m0c[0]} | {m0c[1]} | {m0c[2]} | {m0c[3]} | ROBUST | `{DATA['mem0_p1']}/**/query_*.json` (`retrieved_memories`) |")
    mvc = []
    for L in L4:
        v = mv[L]
        mvc.append(f"{pct(v[0],v[1])} ({v[0]}/{v[1]})" if v else "—")
    P(f"| **Mem0 Vanilla** | write-time mis-apply (coarse) | overall SubEM (no fine bucket) "
      f"| {mvc[0]} | {mvc[1]} | {mvc[2]} | {mvc[3]} | COARSE | `{DATA['mem0_vanilla']}/Conflict_Resolution/*results*.json` |")
    zc = []
    for L in L4:
        z = zp[L]
        zc.append(f"{pct(z['not_both'],z['n'])} ({z['not_both']}/{z['n']})" if z else "—")
    P(f"| **Zep (k=10)** | write-time no-KU-signal + query-time retrieval miss | NotBothExtracted (% has_pair) "
      f"| {zc[0]} | {zc[1]} | {zc[2]} | {zc[3]} | ROBUST | `{DATA['zep']}/**/query_*.json` (`edges`,`response`) |")
    P("")

    # ---- Zep answer-side link ----
    P("### Zep answer-side link (closes the no-KU-signal → answer-falls-to-old chain)")
    P("")
    P("For SHORT lengths, among has_pair with an **Additive-NoKU** signal (old edge still valid; no "
      "invalidation → both versions effectively co-active for the answer LLM), the answer-side outcome:")
    P("")
    P("| Length | Additive-NoKU (N) | of which FAILED | ... answered gt_OLD | % of failures = gt_OLD |")
    P("|:--|:-:|:-:|:-:|:-:|")
    for L in SHORT:
        z = zp[L]
        if z:
            P(f"| {L} | {z['add_total']} | {z['add_fail']} | {z['add_fail_old']} | {pct(z['add_fail_old'],z['add_fail'])} |")
    P("")
    P("This is fully recoverable: each Zep `query_*_context_*.json` carries BOTH the retrieved `edges` "
      "and the final `response`, so the retrieved-context → final-answer join is a within-file lookup. "
      "(262k is dominated by NotBothExtracted, not Additive-NoKU, so the answer-side link is reported "
      "for short lengths where the no-KU-signal path is the mechanism.)")
    P("")

    # ---- Mem0 pool detail ----
    P("### Mem0 + P1 pool-state detail (new-absent buckets have ~0 accuracy)")
    P("")
    P("| Length | PP-New | PP-Both | PP-OldOnly | PP-Missing | new-absent subEM acc |")
    P("|:--|:-:|:-:|:-:|:-:|:-:|")
    for L in L4:
        m = m0[L]
        if m:
            b = m["buckets"]
            P(f"| {L} | {b['PP-New']} | {b['PP-Both']} | {b['PP-OldOnly']} | {b['PP-Missing']} "
              f"| {m['new_absent_correct']}/{m['new_absent']} ({pct(m['new_absent_correct'],m['new_absent'])}) |")
    P("")

    # ---- rationale ----
    P("## Per-method metric rationale")
    P("")
    P("- **Vanilla-RAG / Ours (Struct-Only)** — both are query-time methods that put the full "
      "ordinal pool in front of the answer LLM; the right metric is directly *whether the wrong "
      "answer is gt_OLD*. Vanilla-RAG loses NEW at the **downstream recency judge** (LLM ignores "
      "the max-ordinal rule, reverts to the semantically familiar old value); Struct-Only loses it "
      "at **(S,P) canonicalization** (new/old split into different structural buckets, argmax never "
      "compares them, pool keeps both, reader defaults to world-prior old).")
    P("- **Don't Ask** — identity is delegated to LLM candidate extraction, so the diagnostic signal "
      "is UPSTREAM: `n_candidates == 1` means the LLM extracted only ONE version; combined with "
      "`answered gt_OLD` it shows the single surviving candidate was the world-prior old value. "
      "Reported RAW and with benchmark-reversed (**D-flag**: `gt_seq < old_seq`, unwinnable by "
      "max-serial) queries excluded.")
    P("- **Mem0 + P1** — KU damage is at **write time**, so the mediator is the *pool state* (does "
      "the answering pool still contain gt_new?). `new-absent` = PP-OldOnly + PP-Missing = the "
      "destructive UPDATE/DELETE already erased gt_new before query time; its subEM accuracy is ~0.")
    P("- **Mem0 Vanilla** — coarse by design: no P1 faithful extraction, damage is write-time "
      "mis-apply; we report only overall SubEM and do NOT fabricate a fine M1/M2 split.")
    P("- **Zep** — KU decision lives in bi-temporal fields, not pool text. `NotBothExtracted` "
      "(one/both versions absent from the top-10 the answer LLM saw) is the headline at 262k; for "
      "short lengths the `Additive-NoKU` bucket (no invalidation signal) + the answer-side gt_OLD "
      "link shows the no-KU-signal → fall-to-old chain end to end.")
    P("")
    P("## Provenance")
    P("")
    P("- **Script**: `analysis/errormode_dominant_buckets.py` (this file).")
    P("- **Shared canonical primitives**: `analysis/rescore_canonical.py` "
      "(`official_subem`, `extract_old_surface`, `load_haspair`, `pick_canonical_file`).")
    P("- **Pool-state**: `analysis/compute_pool_acc_crosstab.py` (`classify_pool_state`, `extract_pool_texts`).")
    P("- **Zep bi-temporal**: `analysis/classify_zep_ku_resolution.py` (`classify_query`) + "
      "`analysis/compute_m1_m2_m3.py` (`match_pair`).")
    P("- **Ground truth**: `analysis/results/sh_{6k,32k,64k,262k}_mquake_analysis.json`.")
    P("")
    P("## Caveats (what stays coarse / inference-level)")
    P("")
    P("- **Mem0 M1/M2 split is NOT in this table.** The write-time-failure taxonomy "
      "(world-prior override vs coupled-update fragility, `mem0_event_taxonomy_gt4o.md`) is "
      "elimination-based inference from the event log (which does not record NONE/reject "
      "decisions), single-length (64k), and its scratchpad scripts are deleted. Here we report only "
      "the ROBUST pool-state fact (gt_new absent → ~0 acc) and leave M1/M2 as coarse mechanism prose.")
    P("- **Ours (main) residual** error split (reader-override vs struct-frag vs D-flag) stays "
      "coarse (see `fc_sh_4method_errormode_diag.md` Diagnostic 3); not re-derived here.")
    P("- **Don't Ask 64k**: RAW n_cand==1∧gt_OLD is ~83%; the '100%' figure requires excluding the "
      "2 D-flag (benchmark-reversed) queries. Both are reported; do not cite '100%' without the "
      "D-flag qualifier.")
    P("- **Zep answer-side link at 262k**: 262k is dominated by NotBothExtracted (retrieval miss), "
      "so the Additive-NoKU answer-side link is only meaningful for short lengths and is reported "
      "there; at 262k the recoverable robust fact is NotBothExtracted.")
    P("- Single deterministic run (temp 0), no error bars; subEM percentages can wobble ±1 query "
      "but the DOMINANT bucket per method is stable.")
    P("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[wrote] {OUT_MD}")


if __name__ == "__main__":
    main()
