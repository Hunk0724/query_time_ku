"""FC-SH error-mode + retrieval diagnostics for OURS, from the run's saved
per-query JSONs (actual raw-q retrieval + phase2 resolution; no recompute).
Also loads the HF pool to flag benchmark gold errors (gold = smaller serial,
violating FC's 'larger serial = answer' rule -> unwinnable).
  Usage: python diag_fc_errormodes.py <L>   (L = 6k|32k|64k|262k)
"""
import json, glob, re, sys, os
from collections import Counter

L = sys.argv[1] if len(sys.argv) > 1 else "32k"
OUT = __import__("os").environ.get("OUTPUT_ROOT", "outputs")
ROOT = __import__("os").environ.get("REPO_ROOT") or str(__import__("pathlib").Path(__file__).resolve().parents[1])
QDIR = f"{ROOT}/{OUT}/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified/k_100/factconsolidation_sh_{L}/chunksize_512"
gtmap = {r["query_id"]: r for r in json.load(open(f"{ROOT}/analysis/results/sh_{L}_mquake_analysis.json")) if "query_id" in r}
res = {}
for fs in glob.glob(f"{ROOT}/{OUT}/gpt-4o-mini-mem0-chunk512-temp0-openai-unified/**/*sh_{L}*results*.json", recursive=True):
    for r in json.load(open(fs))["data"]:
        res[r["query_id"]] = r

def norm(t): return re.sub(r"\s+", " ", (t or "").strip().rstrip(".").lower())

# ---- serial map from HF pool (offline); pick the SH row by best gt-text match ----
serial = {}
try:
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    gt_norms = {norm(g["gt_fact_text"]) for g in gtmap.values() if g.get("conflict_type") == "has_pair"}
    best, best_hits = None, -1
    for r in ds:
        lines = re.findall(r"(\d+)\.\s+([^\n]+)", r["context"])
        sm = {norm(t): int(sn) for sn, t in lines}
        hits = sum(1 for g in gt_norms if g in sm)
        if hits > best_hits:
            best_hits, best = hits, sm
    serial = best or {}
except Exception as e:
    print(f"(serial check skipped: {e})")

def load_q(qid):
    fs = glob.glob(f"{QDIR}/query_{qid}_context_*.json")
    return json.load(open(fs[0])) if fs else None

hp = {"both": 0, "new_only": 0, "old_only": 0, "neither": 0}
sp_pair = {"same_sp": 0, "diff_sp": 0, "missing_triple": 0}
resolved = {"clean": 0, "both_kept": 0, "old_kept": 0, "new_missing": 0}
wrong, bench_err = [], []
n_hp = 0
for qid, g in gtmap.items():
    if g.get("conflict_type") != "has_pair" or not g.get("matched"):
        continue
    q = load_q(qid)
    if not q:
        continue
    n_hp += 1
    new_t, old_t = norm(g["gt_fact_text"]), norm(g["old_fact_text"])
    # benchmark-error: gold serial < distractor serial (violates FC rule)
    sn_, so_ = serial.get(new_t), serial.get(old_t)
    is_bench_err = sn_ is not None and so_ is not None and sn_ < so_
    rm = q["retrieved_memories"]
    nh = next((m for m in rm if new_t in norm(m["memory"])), None)
    oh = next((m for m in rm if old_t in norm(m["memory"])), None)
    hp[("both" if nh and oh else "new_only" if nh else "old_only" if oh else "neither")] += 1
    if nh and oh:
        def sp(m):
            t = (m.get("metadata") or {}).get("triple")
            return (t["subject_id"], t["predicate_norm"]) if t else None
        sn, so = sp(nh), sp(oh)
        sp_pair["missing_triple" if (sn is None or so is None) else "same_sp" if sn == so else "diff_sp"] += 1
    ctx = norm(q.get("memories_str", ""))
    nin, oin = new_t in ctx, old_t in ctx
    resolved["clean" if (nin and not oin) else "both_kept" if (nin and oin) else "old_kept" if oin else "new_missing"] += 1
    if not res.get(qid, {}).get("exact_match"):
        if is_bench_err:
            bench_err.append(qid)
        else:
            cause = ("retrieval:new_miss" if not nh else "grouping/resolution:old_not_dropped" if oin
                     else "resolution:new_dropped" if not nin else "answer-LLM")
            wrong.append((qid, cause))

def pct(d, n): return "  ".join(f"{k}={v}({v/max(1,n)*100:.0f}%)" for k, v in d.items())
em = sum(1 for qid in gtmap if gtmap[qid].get("conflict_type") == "has_pair" and gtmap[qid].get("matched") and res.get(qid, {}).get("exact_match"))
print(f"\n=== FC-SH {L} OURS has_pair diagnostics (n={n_hp}, EM={em}/{n_hp}={em/n_hp*100:.0f}%) ===")
print(f"D1 retrieval@100:  {pct(hp,n_hp)}  | recoverable={(hp['both']+hp['new_only'])/n_hp*100:.0f}%")
print(f"D2 (S,P) pairing (both n={hp['both']}):  {pct(sp_pair,hp['both'])}")
print(f"D2b resolution:    {pct(resolved,n_hp)}")
print(f"D3 benchmark-errors (gold<distractor serial, unwinnable): {len(bench_err)}  qids={bench_err}")
print(f"D3 REAL method errors ({len(wrong)}):  {dict(Counter(c for _,c in wrong))}")
print(f"   real-error qids: {[q for q,_ in wrong]}")
print(f"   -> winnable has_pair = {em}/{n_hp-len(bench_err)} = {em/max(1,n_hp-len(bench_err))*100:.0f}% (excl. benchmark errors)")
