#!/usr/bin/env python3
"""gt_new in pool across the FOUR memory-bank configurations (FC-SH 6k, has_pair).

Mirrors paper Table tab:retrieval_len (gpt-4o-mini) but for the 8 LOCAL backbones.
Each config has a DIFFERENT bank, so a single recall number per config is not
enough; this reports gt_new-in-pool per (config x backbone).

Configs:
  1 faithful   ours / Don't Ask / Vanilla-RAG   (ours P1, all versions kept)   top-100
  2 Mem0+FE    faithful P1 + destructive update  (write-time overwrite)         top-100
  3 Zep        bi-temporal graph (held-fixed gpt-4o-mini graph, path A)         top-10
  4 native     Mem0 Vanilla (native extraction + destructive)                   top-100 (gemma3 only)

Reading: for EVERY config, retrieval is not the bottleneck. Misses are either
extraction (config 1, weak backbones) or write-time destructive removal
(config 2/4); Zep = 100% (graph held fixed). See RESULTS.md.

Run:  conda activate "${CONDA_ENV:-repro}" && python analysis/recall_audit_6k/pool_recall_4config_6k.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "analysis"))

from analysis.rescore_canonical import load_haspair
from analysis.compute_pool_acc_crosstab import classify_pool_state

BB = ["gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b",
      "gemma2-9b", "llama3.1-8b", "qwen2.5-7b", "mistral-7b"]

# (label, slim-pool file, is config-1 schema[retrieved_memories dicts] or lean[pool_texts])
CFG = [
    ("1 faithful (ours/DontAsk/Vanilla, top-100)", "pools_slim_6k_no_p5.json", "rm"),
    ("2 Mem0+FE (faithful+destructive, top-100)",  "pools_slim_6k_dest.json",  "pt"),
    ("3 Zep (bi-temporal, top-10)",                "pools_slim_6k_zep.json",   "pt"),
    ("4 Mem0 Vanilla (native, top-100)",           "pools_slim_6k_native.json","pt"),
]


def pool_texts(rec, schema):
    if schema == "rm":
        return [m.get("memory", "") for m in rec.get("retrieved_memories", [])]
    return rec.get("pool_texts", [])


def main():
    hp = load_haspair("6k")
    print(f"gt_new in pool  (FC-SH 6k has_pair N={len(hp)})\n")
    print(f"{'config':<44}" + "".join(f"{b.split('-')[0][0]+b.split('-')[1]:>8}" for b in BB))
    for label, fn, schema in CFG:
        pools = json.load(open(os.path.join(HERE, fn), encoding="utf-8"))
        cells = []
        for bb in BB:
            pb = pools.get(bb)
            if not pb:
                cells.append("    -   "); continue
            n = new = 0
            for qid, row in hp.items():
                rec = pb.get(str(qid))
                if rec is None:
                    continue
                st = classify_pool_state(pool_texts(rec, schema),
                                         row.get("gt_fact_text") or "",
                                         row.get("old_fact_text") or "")
                n += 1; new += st in ("PP-New", "PP-Both")
            cells.append(f"{100*new/n:>6.0f}% " if n else "    -   ")
        print(f"{label:<44}" + "".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
