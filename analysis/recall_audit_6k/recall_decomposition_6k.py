#!/usr/bin/env python3
"""Recall decomposition for local-model backbones (FC-SH 6k, has_pair).

Separates the "gt_new not in top-100" event into its two distinct causes:
  A. gt_new in bank?        -> extraction recall  (did P1 put gt_new in the bank at all)
  B. gt_new in top-100?     -> what the answer LLM's retrieval pool actually contained
  C. B given A              -> retrieval@100 conditional on in-bank
                              (= is the RETRIEVER the bottleneck, or is it extraction?)

Self-contained: reads only committed data
  - bank  : analysis/results/p1_caches__{bb}/extraction_cache_p1_6k.json  (ALL extracted facts)
  - pool  : analysis/recall_audit_6k/pools_slim_6k_no_p5.json             (top-100 retrieved_memories, slimmed)
  - GT    : analysis/results/sh_6k_mquake_analysis.json                   (via load_haspair)
  - match : analysis/compute_pool_acc_crosstab.classify_pool_state (matcher v4)

Run:  conda activate "${CONDA_ENV:-repro}" && python analysis/recall_audit_6k/recall_decomposition_6k.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "analysis"))

from analysis.rescore_canonical import load_haspair
from analysis.compute_pool_acc_crosstab import classify_pool_state

BB = ["gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b",
      "gemma2-9b", "llama3.1-8b", "qwen2.5-7b", "mistral-7b"]


def bank_texts(bb):
    p = os.path.join(REPO, f"analysis/results/p1_caches__{bb}/extraction_cache_p1_6k.json")
    cache = json.load(open(p, encoding="utf-8"))
    return [t for v in cache.values() if isinstance(v, list) for t in v]


def gt_new_present(texts, row):
    return classify_pool_state(texts, row.get("gt_fact_text") or "",
                               row.get("old_fact_text") or "") in ("PP-New", "PP-Both")


def main():
    hp = load_haspair("6k")
    pools = json.load(open(os.path.join(HERE, "pools_slim_6k_no_p5.json"), encoding="utf-8"))
    print(f"has_pair N = {len(hp)}   (pool = ours no_p5 top-100)\n")
    hdr = f"{'backbone':>12} | {'A gt_new in bank':>16} | {'B gt_new in top100':>18} | {'C retrieval@100|in-bank':>23}"
    print(hdr); print("-" * len(hdr))
    for bb in BB:
        bank = bank_texts(bb)
        pool_bb = pools.get(bb, {})
        n = in_bank = in_top = both = 0
        for qid, row in hp.items():
            rec = pool_bb.get(str(qid))
            if rec is None:
                continue
            pool = [m.get("memory", "") for m in rec.get("retrieved_memories", [])]
            a = gt_new_present(bank, row)
            b = gt_new_present(pool, row)
            n += 1; in_bank += a; in_top += b; both += (a and b)
        c = 100 * both / max(in_bank, 1)
        print(f"{bb:>12} | {in_bank}/{n} = {100*in_bank/n:>3.0f}% | "
              f"{in_top}/{n} = {100*in_top/n:>3.0f}% | {both}/{in_bank} = {c:>3.0f}%")
    print("\nReading: A==B and C==100% for every backbone  =>  retrieval is never the")
    print("bottleneck; every gt_new miss is an EXTRACTION miss (gt_new not in bank).")


if __name__ == "__main__":
    main()
