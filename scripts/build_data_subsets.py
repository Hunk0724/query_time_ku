#!/usr/bin/env python3
"""Rebuild the in-repo dataset subsets from their official public sources.

The repo already ships these subsets under `data/` so experiments run offline;
this script documents exactly how they were derived and lets anyone regenerate
them from the official releases.

Outputs (paper-relevant subsets only):
  data/longmemeval/longmemeval_s_ku.json
      = the 78 `knowledge-update` questions of LongMemEval-S (cleaned).
        Official source: HF dataset `xiaowu0162/longmemeval-cleaned`
        (`longmemeval_s_cleaned.json`, 500 questions), MIT, Wu et al. 2024.
  data/fc_sh/Conflict_Resolution_factconsolidation_sh.json
      = the 4 FC-SH FactConsolidation-Single-Hop rows (6k/32k/64k/262k) of the
        Conflict_Resolution split. Official source: HF dataset
        `ai-hyz/MemoryAgentBench`, MIT, Hu et al. 2026.

Usage:
  python scripts/build_data_subsets.py            # rebuild both
  python scripts/build_data_subsets.py --lme-src /path/to/longmemeval_s_cleaned.json
"""
import argparse, json, os

def build_lme(src, out):
    data = json.load(open(src))
    ku = [q for q in data if q.get("question_type") == "knowledge-update"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(ku, open(out, "w"), ensure_ascii=False)
    print(f"[lme] {len(ku)} knowledge-update questions -> {out} "
          f"({os.path.getsize(out)/1e6:.1f} MB)")

def build_fc(out_dir):
    # One file per context length, keyed by the `metadata.source` value that the
    # dataset config's `sub_dataset` field selects (factconsolidation_sh_<L>) —
    # mirrors how each length is run as its own config.
    from datasets import load_dataset
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
    os.makedirs(out_dir, exist_ok=True)
    for r in ds:
        src = (r.get("metadata") or {}).get("source", "")
        if not src.startswith("factconsolidation_sh_"):
            continue
        out = os.path.join(out_dir, f"{src}.json")
        json.dump([dict(r)], open(out, "w"), ensure_ascii=False)
        print(f"[fc-sh] {src} -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lme-src", default="data/longmemeval/longmemeval_s_cleaned.json",
                    help="official longmemeval_s_cleaned.json (clone "
                         "huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)")
    ap.add_argument("--lme-out", default="data/longmemeval/longmemeval_s_ku.json")
    ap.add_argument("--fc-out", default="data/fc_sh")
    ap.add_argument("--skip-lme", action="store_true")
    ap.add_argument("--skip-fc", action="store_true")
    a = ap.parse_args()
    if not a.skip_lme:
        if os.path.exists(a.lme_src):
            build_lme(a.lme_src, a.lme_out)
        else:
            print(f"[lme] official source not found at {a.lme_src}; "
                  f"clone huggingface.co/datasets/xiaowu0162/longmemeval-cleaned first")
    if not a.skip_fc:
        build_fc(a.fc_out)
