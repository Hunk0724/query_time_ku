"""Zep-specific KU-resolution classifier (bi-temporal mediator).

Motivation (see matcher_specification.md §3.4):
    The text-presence pool_state (classify_pool_state in
    compute_pool_acc_crosstab.py) is the correct *mediator* for mem0 / ours,
    whose KU resolution lands in the pool TEXT (mem0 deletes at write-time; ours
    resolves at query-time to a clean new-only pool). For Zep it is NOT: Zep's
    search returns invalid edges too, so the pool is ~always PP-Both and text
    presence carries no KU signal. Zep encodes its KU decision in the
    BI-TEMPORAL fields (valid_at / invalid_at / expired_at), deferring the final
    judgment to the answer LLM that reads the date range.

    This script reads those fields and classifies each has_pair query into what
    Zep ACTUALLY did with the (gt_new, gt_old) pair, then crosses it with EM.

Buckets (per has_pair query, over the top-10 edges the answer LLM saw).
IMPORTANT — the two "Resolved" buckets require a VERIFIED contradiction handoff,
not merely "some matching edge has invalid_at". graphiti sets the superseded
edge's invalid_at EXACTLY to the winner's valid_at, so a genuine pairwise KU
between THESE two versions shows old.invalid_at == new.valid_at (or the mirror).
Without that check, an edge invalidated by a THIRD fact (or a matcher over-match
on a generic (S,P) stem shared across entities) would be miscounted as a
backward/forward resolution. Those non-handoff cases go to Other-Ambiguous.

    Resolved-Correct   : both versions present; EXISTS an OLD-matching edge whose
                         invalid_at == a NEW-matching edge's valid_at. Zep
                         superseded old in favour of new (correct KU).
    Resolved-Backward  : both present; EXISTS a NEW-matching edge whose
                         invalid_at == an OLD-matching edge's valid_at. Zep
                         invalidated the newer (counterfactual) fact —
                         world-prior mis-resolution.
    Additive-NoKU      : both present; ALL matching edges (both versions) have
                         invalid_at is None. Zep never detected the conflict →
                         two co-active facts dumped on the answer LLM with no
                         disambiguating temporal signal.
    Other-Ambiguous    : both present; some invalid_at is set but forms NO clean
                         pairwise handoff between the two versions (invalidated
                         by a 3rd fact, both-invalid, one-sided missing valid_at,
                         multi-edge / generic-stem matcher over-match). Residual
                         tail — report %, do not over-interpret.
    NotBothExtracted   : one or both versions absent from top-10 edges
                         (retrieval / extraction miss, not a KU decision).

Canonical deps (run under the repro env so nltk / MAB eval utils import):
    match_pair            <- analysis/compute_m1_m2_m3.py
    load_gt, _em_from_perqid <- analysis/compute_pool_acc_crosstab.py

Usage:
    python analysis/classify_zep_ku_resolution.py            # 6k,32k,64k
    python analysis/classify_zep_ku_resolution.py --length 64k
    python analysis/classify_zep_ku_resolution.py --dump-qids   # per-qid table
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

from compute_m1_m2_m3 import match_pair  # noqa: E402
from compute_pool_acc_crosstab import load_gt, _em_from_perqid  # noqa: E402

ZEP_ROOT = REPO / "outputs/rag_retrieved/Structure_rag_zep/k_10"
LENGTHS = ["6k", "32k", "64k", "262k"]

BUCKETS = [
    "Resolved-Correct",
    "Resolved-Backward",
    "Additive-NoKU",
    "Other-Ambiguous",
    "NotBothExtracted",
]


def _match_edges(edges, gt_new, gt_old, target):
    """ALL edges whose .fact matches the target version (new|old)."""
    return [e for e in edges if match_pair(e.get("fact", ""), gt_new, gt_old, target)]


def _handoff(loser_edges, winner_edges):
    """True iff some loser edge's invalid_at == some winner edge's valid_at
    (graphiti's exact contradiction handoff between THESE two versions)."""
    winner_valids = {e.get("valid_at") for e in winner_edges if e.get("valid_at")}
    for le in loser_edges:
        iv = le.get("invalid_at")
        if iv and iv in winner_valids:
            return True
    return False


def classify_query(edges, gt_new, gt_old):
    """Return (bucket, detail dict). Resolved-* require a verified handoff."""
    ens = _match_edges(edges, gt_new, gt_old, "new")
    eos = _match_edges(edges, gt_new, gt_old, "old")
    if not (ens and eos):
        return "NotBothExtracted", {
            "new_present": bool(ens), "old_present": bool(eos)}

    detail = {
        "n_new_edges": len(ens), "n_old_edges": len(eos),
        "new_invalids": [e.get("invalid_at") for e in ens],
        "old_invalids": [e.get("invalid_at") for e in eos],
    }
    any_inval = any(e.get("invalid_at") for e in ens + eos)

    # Verified pairwise handoffs (winner's valid_at == loser's invalid_at)
    correct = _handoff(loser_edges=eos, winner_edges=ens)   # old superseded by new
    backward = _handoff(loser_edges=ens, winner_edges=eos)  # new superseded by old

    if correct and not backward:
        return "Resolved-Correct", detail
    if backward and not correct:
        return "Resolved-Backward", detail
    if not any_inval:
        return "Additive-NoKU", detail
    # invalid_at set somewhere but no clean single-direction handoff, OR both
    # directions fire (3rd-fact invalidation, both-invalid, matcher over-match)
    return "Other-Ambiguous", detail


def analyze_length(L, dump_qids=False):
    gt = load_gt(L)
    hp_qids = sorted(q for q, e in gt.items()
                     if e.get("conflict_type") == "has_pair")
    qdir = ZEP_ROOT / f"factconsolidation_sh_{L}/chunksize_512"

    bucket_ct = Counter()
    bucket_em = Counter()          # (bucket, em_bool)
    n_multi_edge = 0               # queries with >1 edge matching a version
    n_no_json = 0
    rows = []

    # Global invalidation coverage (over ALL 100 queries, not only has_pair — the
    # 6.3%-style headline is 'of all edges the answer LLM sees, how many carry
    # invalid_at'). Sample space = every edge in every query_*.json across the
    # full FC-SH test set at this length.
    all_qids = sorted(gt.keys())
    n_edges_total = 0
    n_edges_invalidated = 0
    n_queries_read = 0

    for qid in all_qids:
        files = glob.glob(str(qdir / f"query_{qid}_context_*.json"))
        if not files:
            continue
        try:
            j = json.load(open(files[0], encoding="utf-8"))
        except Exception:
            continue
        n_queries_read += 1
        for e in (j.get("edges") or []):
            n_edges_total += 1
            if e.get("invalid_at"):
                n_edges_invalidated += 1

    for qid in hp_qids:
        files = glob.glob(str(qdir / f"query_{qid}_context_*.json"))
        if not files:
            n_no_json += 1
            continue
        try:
            j = json.load(open(files[0], encoding="utf-8"))
        except Exception:
            n_no_json += 1
            continue
        edges = j.get("edges", []) or []
        gt_new = gt[qid].get("gt_fact_text") or ""
        gt_old = gt[qid].get("old_fact_text") or ""
        bucket, detail = classify_query(edges, gt_new, gt_old)
        em = _em_from_perqid(j.get("response") or "", gt[qid].get("gt_answer"))

        bucket_ct[bucket] += 1
        bucket_em[(bucket, em)] += 1
        if detail.get("n_new_edges", 1) > 1 or detail.get("n_old_edges", 1) > 1:
            n_multi_edge += 1
        if dump_qids:
            rows.append((qid, bucket, em, detail))

    n = sum(bucket_ct.values())
    return {
        "length": L, "n_hp": len(hp_qids), "n_classified": n,
        "n_no_json": n_no_json, "n_multi_edge": n_multi_edge,
        "bucket_ct": bucket_ct, "bucket_em": bucket_em,
        "rows": rows,
        "n_queries_read": n_queries_read,
        "n_edges_total": n_edges_total,
        "n_edges_invalidated": n_edges_invalidated,
    }


def print_report(res):
    L = res["length"]
    n = res["n_classified"]
    print(f"\n{'='*66}\nZep KU-resolution — {L}  (has_pair N={res['n_hp']}, "
          f"classified={n}, no_json={res['n_no_json']})\n{'='*66}")
    print(f"{'bucket':<20}{'n':>5}{'pct':>7}   "
          f"{'EM✓':>5}{'EM✗':>5}{'acc%':>7}")
    for b in BUCKETS:
        c = res["bucket_ct"].get(b, 0)
        ok = res["bucket_em"].get((b, True), 0)
        no = res["bucket_em"].get((b, False), 0)
        pct = 100 * c / n if n else 0
        acc = 100 * ok / c if c else 0
        print(f"{b:<20}{c:>5}{pct:>6.1f}%   {ok:>5}{no:>5}{acc:>6.1f}%")
    tot_ok = sum(v for (b, em), v in res["bucket_em"].items() if em)
    print(f"{'-'*50}\n{'TOTAL':<20}{n:>5}{'':>7}   "
          f"{tot_ok:>5}{n - tot_ok:>5}{100*tot_ok/n if n else 0:>6.1f}%")
    print(f"\n  note: Resolved-* are handoff-verified (winner.valid==loser.invalid). "
          f"multi-edge-per-version queries: {res['n_multi_edge']}/{n}")
    # Invalidation coverage — over all edges the answer LLM sees across the full
    # test set (all 100 queries), not restricted to has_pair. This is the
    # 6.3%-style headline for "how often does Zep tag an edge as invalid".
    te = res.get("n_edges_total", 0)
    ie = res.get("n_edges_invalidated", 0)
    qr = res.get("n_queries_read", 0)
    if te:
        print(f"  invalidation coverage @ {L}: {ie}/{te} edges = "
              f"{100*ie/te:.2f}% carry invalid_at "
              f"(across {qr} queries, avg {te/qr:.1f} edges/query)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", choices=LENGTHS, default=None)
    ap.add_argument("--dump-qids", action="store_true")
    args = ap.parse_args()
    lengths = [args.length] if args.length else LENGTHS

    all_res = []
    for L in lengths:
        res = analyze_length(L, dump_qids=args.dump_qids)
        all_res.append(res)
        print_report(res)
        if args.dump_qids:
            print(f"\n  per-qid ({L}):")
            for qid, bucket, em, d in res["rows"]:
                tag = "EM✓" if em else "EM✗"
                mult = (f" [multi n={d.get('n_new_edges')}/{d.get('n_old_edges')}]"
                        if d.get("n_new_edges", 1) > 1 or d.get("n_old_edges", 1) > 1
                        else "")
                print(f"    qid={qid:<4} {bucket:<18} {tag}{mult}")

    # compact machine-readable summary
    print(f"\n{'='*66}\nSUMMARY (bucket %)\n{'='*66}")
    print(f"{'length':<8}" + "".join(f"{b.split('-')[0][:8]:>10}"
                                     for b in BUCKETS) + f"{'acc%':>8}")
    for res in all_res:
        n = res["n_classified"]
        line = f"{res['length']:<8}"
        for b in BUCKETS:
            c = res["bucket_ct"].get(b, 0)
            line += f"{(100*c/n if n else 0):>9.0f}%"
        tot_ok = sum(v for (b, em), v in res["bucket_em"].items() if em)
        line += f"{(100*tot_ok/n if n else 0):>7.0f}%"
        print(line)


if __name__ == "__main__":
    main()
