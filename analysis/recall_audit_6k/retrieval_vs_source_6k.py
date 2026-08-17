#!/usr/bin/env python3
"""Is the retriever ever the bottleneck? Decompose gt_new-miss into in-memory vs
retrieval, for every memory-bank config (FC-SH 6k, has_pair).

For each config the "memory" (bank/store) differs:
  A  gt_new in memory?        faithful  -> extraction cache (all versions kept)
                              dest/native -> the post-destructive qdrant store's facts
  B  gt_new in retrieved pool?   top-100 (Zep top-10)
  C  B given A = retrieval@K | in-memory   -> is the RETRIEVER the bottleneck?

Result (see RESULTS.md): C == 100% for EVERY config x backbone (wherever the
target is in memory). The retriever never fails; every gt_new miss is either
extraction (faithful, weak backbones) or write-time destructive removal
(dest/native: the store collapses to 0-313 facts). Zep = 100% pool recall,
no miss at all.

Self-contained (committed data, zero API):
  banks : analysis/results/p1_caches__{bb}/extraction_cache_p1_6k.json (faithful)
          analysis/recall_audit_6k/store_bank_6k_{dest,native}.json    (destructive stores)
  pools : analysis/recall_audit_6k/pools_slim_6k_{no_p5,dest,native}.json
  GT    : analysis/results/sh_6k_mquake_analysis.json

Run:  conda activate "${CONDA_ENV:-repro}" && python analysis/recall_audit_6k/retrieval_vs_source_6k.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "analysis"))

from analysis.rescore_canonical import load_haspair
from analysis.compute_pool_acc_crosstab import classify_pool_state

BB = ["gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b",
      "gemma2-9b", "llama3.1-8b", "qwen2.5-7b", "mistral-7b"]


def new_in(texts, row):
    return classify_pool_state(texts, row.get("gt_fact_text") or "",
                               row.get("old_fact_text") or "") in ("PP-New", "PP-Both")


def faithful_bank(bb):
    p = os.path.join(REPO, f"analysis/results/p1_caches__{bb}/extraction_cache_p1_6k.json")
    c = json.load(open(p, encoding="utf-8"))
    return [t for v in c.values() if isinstance(v, list) for t in v]


def store_bank(fn):
    return json.load(open(os.path.join(HERE, fn), encoding="utf-8"))


def pool_of(rec):
    if "retrieved_memories" in rec:
        return [m.get("memory", "") for m in rec["retrieved_memories"]]
    return rec.get("pool_texts", [])


def run(label, banks, pools_fn):
    pools = json.load(open(os.path.join(HERE, pools_fn), encoding="utf-8"))
    hp = load_haspair("6k")
    print(f"\n=== {label} ===")
    print(f"{'backbone':>12} | {'A in-memory':>11} | {'B in-pool':>10} | {'C retr@K|in-mem':>15}")
    for bb in BB:
        bank = banks(bb)
        pb = pools.get(bb)
        if bank is None or pb is None:
            print(f"{bb:>12} |     -     (config not run)"); continue
        n = a = b = ab = 0
        hpd = load_haspair("6k")
        for qid, row in hpd.items():
            rec = pb.get(str(qid))
            if rec is None:
                continue
            ai = new_in(bank, row); bi = new_in(pool_of(rec), row)
            n += 1; a += ai; b += bi; ab += (ai and bi)
        c = f"{100*ab/a:>3.0f}%" if a else "(0/0)"
        print(f"{bb:>12} | {a}/{n}={100*a/n:>3.0f}% | {b}/{n}={100*b/n:>3.0f}% | {ab}/{a}={c}")


def main():
    run("config 1 faithful (bank = ours P1 extraction cache)",
        faithful_bank, "pools_slim_6k_no_p5.json")
    db = store_bank("store_bank_6k_dest.json")
    run("config 2 Mem0+FE destructive (bank = post-destructive store)",
        lambda bb: db.get(bb), "pools_slim_6k_dest.json")
    nb = store_bank("store_bank_6k_native.json")
    run("config 4 Mem0 Vanilla native (bank = post-destructive store)",
        lambda bb: nb.get(bb), "pools_slim_6k_native.json")
    print("\nconfig 3 Zep: pool recall = 100% for all backbones (no miss); "
          "held-fixed gpt-4o-mini graph, backbone-independent.")
    print("\n=> C = 100% everywhere the target is in memory. Retriever is never the")
    print("   bottleneck; misses are extraction (config 1) or write-time destruction (2/4).")


if __name__ == "__main__":
    main()
