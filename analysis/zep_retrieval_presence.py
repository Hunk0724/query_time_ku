#!/usr/bin/env python3
"""Zep retrieval-presence row for tab:retrieval_len (100/98/100/23).

PURPOSE (paper tab:retrieval_len, Zep row).
    The append-only methods share ONE retrieval whose ceiling is measured by
    analysis/fcsh_retrieval_ceiling.py. Zep retrieves differently (its own
    bi-temporal graph search returning top-10 edges), so its retrieval-presence
    is measured on ITS OWN edge dumps, not the shared vector pool.

    Metric = over has_pair queries, the fraction whose top-10 Zep edges CONTAIN
    the current-version fact (gt_new), using the canonical (S,P)+object matcher
    (match_pair from compute_m1_m2_m3). This isolates RETRIEVAL (did Zep surface
    gt_new at all) from the answer LLM's downstream KU judgment. At 262k Zep's
    retrieval collapses, which is the point the table makes.

Reproduces the paper's Zep row: 6k 100%, 32k 98%, 64k 100%, 262k 23%.

Committed data only; no model re-run; no .env. Reuses canonical primitives.
    OUTPUT_ROOT=reference_outputs python analysis/zep_retrieval_presence.py
"""
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from compute_m1_m2_m3 import match_pair  # noqa: E402
from compute_pool_acc_crosstab import load_gt  # noqa: E402

# Source of run outputs: your own fresh runs ("outputs", default) or the shipped
# reference ("reference_outputs"). Override: OUTPUT_ROOT=reference_outputs python ...
OUT = os.environ.get("OUTPUT_ROOT", "outputs")
ZEP_ROOT = REPO / OUT / "rag_retrieved/Structure_rag_zep/k_10"
LENGTHS = ["6k", "32k", "64k", "262k"]


def presence_at(L):
    """Return (n_gt_new_present, n_both_present, n_has_pair_with_dump)."""
    gt = load_gt(L)
    hp = {q: e for q, e in gt.items() if e.get("conflict_type") == "has_pair"}
    qdir = ZEP_ROOT / f"factconsolidation_sh_{L}/chunksize_512"
    new_p = both_p = tot = 0
    for qid, e in hp.items():
        fs = glob.glob(str(qdir / f"query_{qid}_context_*.json"))
        if not fs:
            continue
        edges = json.load(open(fs[0], encoding="utf-8")).get("edges", []) or []
        gn = e.get("gt_fact_text") or ""
        go = e.get("old_fact_text") or ""
        has_new = any(match_pair(x.get("fact", ""), gn, go, "new") for x in edges)
        has_old = any(match_pair(x.get("fact", ""), gn, go, "old") for x in edges)
        new_p += has_new
        both_p += has_new and has_old
        tot += 1
    return new_p, both_p, tot


def main():
    print("Zep retrieval-presence over has_pair queries "
          f"(source: {OUT}/rag_retrieved/Structure_rag_zep/k_10)")
    print(f"{'length':<8}{'gt_new present':>18}{'both present':>16}")
    for L in LENGTHS:
        new_p, both_p, tot = presence_at(L)
        if not tot:
            print(f"{L:<8}{'(no Zep dumps found)':>18}")
            continue
        print(f"{L:<8}{f'{new_p}/{tot} = {100*new_p/tot:.0f}%':>18}"
              f"{f'{both_p}/{tot} = {100*both_p/tot:.0f}%':>16}")
    print("\ntab:retrieval_len Zep row = the 'gt_new present' column "
          "(paper: 100 / 98 / 100 / 23).")


if __name__ == "__main__":
    main()
