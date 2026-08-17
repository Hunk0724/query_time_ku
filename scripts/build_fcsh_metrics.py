#!/usr/bin/env python3
"""Consolidate FC-SH results into outputs/fcsh_metrics.json.

Parses each mem0-family result file under outputs/ into a clean, unambiguous
row keyed by (backbone, method_tag, length) — the three fields the paper tables
and the compare tool both need. The metric is substring exact match (sEM), the
`substring_exact_match` field averaged over the file's rows.

Covers the methods produced by scripts/run_fc_sh.sh. Zep, LCA and Don't Ask use
separate entry scripts with different output formats; their sEM is reported in
the paper tables directly directly.

Usage: python scripts/build_fcsh_metrics.py [--outputs outputs] [--out outputs/fcsh_metrics.json]
"""
import argparse, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG2METHOD = {
    "unified_no_p5": "ours (main)",
    "unified_struct": "ours (struct)",
    "unified_p3_only_no_struct": "ours (llm-only)",
    "unified_dest": "mem0 + fact extraction",
    "native": "mem0 vanilla",
    "unified_q_llm_recency": "vanilla-rag",
    "unified": "ours (+P5)",
}
DIR_RE = re.compile(
    r"^(?P<prefix>.+?)-mem0-chunk512-temp0-openai-"
    r"(?P<tag>unified_no_p5|unified_struct|unified_p3_only_no_struct|"
    r"unified_dest|native|unified_q_llm_recency|unified)"
    r"(?:__(?P<mt>.+))?$")
LEN_RE = re.compile(r"sh_(262k|64k|32k|6k)")

def sem(path):
    rows = json.load(open(path))["data"]
    return sum(1 for r in rows if r.get("substring_exact_match")), len(rows)

def build(outputs_dir, out):
    rows = []
    for p in glob.glob(os.path.join(outputs_dir, "*", "**", "*results*.json"), recursive=True):
        dir0 = os.path.relpath(p, outputs_dir).split(os.sep)[0]
        m = DIR_RE.match(dir0)
        if not m:
            continue
        lm = LEN_RE.search(os.path.basename(p))
        if not lm:
            continue
        s, n = sem(p)
        rows.append({
            "backbone": m.group("mt") or m.group("prefix"),
            "method_tag": m.group("tag"),
            "method": TAG2METHOD[m.group("tag")],
            "length": lm.group(1),
            "overall_subem": s, "overall_N": n,
            "source_dir": dir0,
        })
    rows.sort(key=lambda r: (r["backbone"], r["method_tag"], r["length"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"{len(rows)} rows -> {out}")
    print("backbones:", sorted(set(r["backbone"] for r in rows)))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "fcsh_metrics.json"))
    a = ap.parse_args()
    build(a.outputs, a.out)
