"""Aggregate a run's cost/latency: per-stage calls, tokens, time, est USD.

Reads the MEM0_COST_LOG jsonl (write-side extraction/triple/grouping/conflict +
embeddings) and optionally the per-query answer JSONs (which already carry the
answer LLM's prompt_tokens/completion_tokens). Usage:
  python summarize_cost.py <cost_log.jsonl> [answer_json_dir]
"""
import json
import sys
import glob
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.cost_logger import PRICE  # noqa: E402


def est_usd(model, pt, ct):
    p = PRICE.get(model)
    return (pt / 1e6 * p["in"] + ct / 1e6 * p["out"]) if p else 0.0


def main():
    cost_log = sys.argv[1]
    ans_dir = sys.argv[2] if len(sys.argv) > 2 else None
    agg = defaultdict(lambda: {"calls": 0, "pt": 0, "ct": 0, "t": 0.0, "model": ""})

    for line in open(cost_log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = agg[r["stage"]]
        a["calls"] += 1
        a["pt"] += r.get("prompt_tokens", 0)
        a["ct"] += r.get("completion_tokens", 0)
        a["t"] += r.get("latency_s", 0.0)
        a["model"] = r.get("model", a["model"])

    # answer LLM tokens from per-query JSONs (already saved by agent.py)
    if ans_dir:
        a = agg["answer_p4"]
        for f in glob.glob(f"{ans_dir}/query_*.json"):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            a["calls"] += 1
            a["pt"] += d.get("prompt_tokens", 0) or 0
            a["ct"] += d.get("completion_tokens", 0) or 0
            a["model"] = "gpt-4o-mini"

    print(f"{'stage':<16}{'calls':>8}{'in_tok':>12}{'out_tok':>10}{'time_s':>10}{'est_$':>9}")
    tot_pt = tot_ct = 0
    tot_t = tot_usd = 0.0
    for stage, a in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
        usd = est_usd(a["model"], a["pt"], a["ct"])
        tot_pt += a["pt"]; tot_ct += a["ct"]; tot_t += a["t"]; tot_usd += usd
        print(f"{stage:<16}{a['calls']:>8}{a['pt']:>12,}{a['ct']:>10,}{a['t']:>10.1f}{usd:>9.3f}")
    print(f"{'TOTAL':<16}{'':>8}{tot_pt:>12,}{tot_ct:>10,}{tot_t:>10.1f}{tot_usd:>9.3f}")
    print(f"\n(LLM wall-clock {tot_t:.0f}s; est cost ${tot_usd:.3f}. "
          f"Note: write-side calls are sequential -> time≈wall; answer/grouping may overlap.)")


if __name__ == "__main__":
    main()
