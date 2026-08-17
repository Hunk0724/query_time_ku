#!/usr/bin/env python3
"""tab:errormode_gtold — of each query-time method's has_pair failures, the share
whose final answer is gt_old (the world-prior old value).

Don't Ask reads its maxserial output (answer + gt_answer + n_candidates); Vanilla-RAG
reads the q_llm_recency result file (output + answer). A failure = not sEM-correct;
gt_old = official_subem(prediction, old_surface). Reuses the canonical primitives.

Source of run outputs: OUTPUT_ROOT (default "outputs"; set "reference_outputs" to
reproduce the paper numbers from the shipped reference without rerunning).
    python analysis/errormode_gtold.py
    OUTPUT_ROOT=reference_outputs python analysis/errormode_gtold.py
"""
import glob, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))
from analysis.rescore_canonical import (   # noqa: E402
    load_haspair, official_subem, extract_old_surface, pick_canonical_file,
)

OUT = os.environ.get("OUTPUT_ROOT", "outputs")
L4 = ["6k", "32k", "64k", "262k"]
DONT_ASK = str(REPO / f"{OUT}/maxserial_theircode/{{L}}_gpt-4o-mini_vector100.json")
VANILLA = str(REPO / f"{OUT}/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_q_llm_recency")


def _old(row):
    return extract_old_surface(row.get("gt_fact_text", ""), row.get("old_fact_text", ""))


def dont_ask_gtold(L):
    p = DONT_ASK.format(L=L)
    if not os.path.exists(p):
        return None
    rows = {r["query_id"]: r for r in json.load(open(p))["rows"]}
    hp = load_haspair(L); nfail = old = 0
    for qid, row in hp.items():
        r = rows.get(qid)
        if r is None or official_subem(r.get("answer"), [r.get("gt_answer")]):
            continue
        nfail += 1
        os_ = _old(row)
        old += bool(os_ and official_subem(r.get("answer"), [os_]))
    return old, nfail


def vanilla_gtold(L):
    f = pick_canonical_file(VANILLA, L)
    if not f:
        return None
    byq = {d["query_id"]: d for d in json.load(open(f))["data"]}
    hp = load_haspair(L); nfail = old = 0
    for qid, row in hp.items():
        d = byq.get(qid)
        if d is None or official_subem(d.get("output"), d.get("answer")):
            continue
        nfail += 1
        os_ = _old(row)
        old += bool(os_ and official_subem(d.get("output"), [os_]))
    return old, nfail


def fmt(t):
    return f"{t[0]}/{t[1]} ({100*t[0]/t[1]:.0f}%)" if t and t[1] else "-"


if __name__ == "__main__":
    print("tab:errormode_gtold - of has_pair failures, answered gt_old (gpt-4o-mini)")
    hdr = "Method"; da = "Don't Ask"; vr = "Vanilla-RAG"
    print(f"{hdr:>12} | " + " ".join(f"{L:>14}" for L in L4))
    print(f"{da:>12} | " + " ".join(f"{fmt(dont_ask_gtold(L)):>14}" for L in L4))
    print(f"{vr:>12} | " + " ".join(f"{fmt(vanilla_gtold(L)):>14}" for L in L4))
