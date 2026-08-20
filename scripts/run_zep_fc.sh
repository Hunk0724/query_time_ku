#!/bin/bash
# Zep baseline on FC-SH via main.py (benchmark flow: build graph -> async wait ->
# graph search). chunk512/temp0 aligned to ours; rest = Zep default.
#   Usage: RUN_OAI_KEY_NAME=OPENAI_API_KEY_C bash run_zep_fc.sh <L>
# NOTE: Zep is a hosted ASYNC service — see README §6 "Running Zep" before use.
#   - Wait knobs: ZEP_INITIAL_WAIT (default 360s; 1200s @262k), ZEP_MAX_PROBES
#     (8; 20 @262k). Do NOT hard-cut the wait; empty retrieval usually = queried
#     before the graph finished indexing.
#   - Retrieval top-k: ZEP_TOP_K (default 10, per Zep official).
#   - FREE-PLAN caps are the usual cause of empty/low results: message content is
#     truncated to 2400 chars (methods/zep.py) and get_user_context is ~5/min
#     (429 backoff). Use a paid plan, rotate ZEP_API_KEY_A/B/C via RUN_ZEP_KEY_NAME,
#     or self-host Graphiti. Anomalously low scores are operational, not a code bug.
set -u
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
LME_DATA_DIR="${LME_DATA_DIR:-$REPO_ROOT/data/longmemeval}"
export LME_DATA="${LME_DATA:-$LME_DATA_DIR/longmemeval_s_cleaned.json}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-query_time_ku}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd $REPO_ROOT
set -a; [[ -f .env ]] && . .env; set +a
[[ -n "${RUN_OAI_KEY_NAME:-}" ]] && export OPENAI_API_KEY="${!RUN_OAI_KEY_NAME}" && echo "[key] \$$RUN_OAI_KEY_NAME ...${OPENAI_API_KEY: -6}"
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[key] ERROR none"; exit 1; }
# Zep key (from https://www.getzep.com/): use a direct ZEP_API_KEY in .env, or pick
# a named key via RUN_ZEP_KEY_NAME (default ZEP_API_KEY_A) for parallel runs.
RUN_ZEP_KEY_NAME="${RUN_ZEP_KEY_NAME:-ZEP_API_KEY_A}"
if [[ -z "${ZEP_API_KEY:-}" ]]; then export ZEP_API_KEY="${!RUN_ZEP_KEY_NAME:-}"; fi
[[ -z "${ZEP_API_KEY:-}" ]] && { echo "[zep-key] ERROR: no ZEP_API_KEY (set it in .env, or point RUN_ZEP_KEY_NAME at a key that exists)"; exit 1; }
echo "[zep-key] using key ...${ZEP_API_KEY: -6}"

L="${1:?need L}"
LOGROOT=logs
DCONF=configs/data_conf/Conflict_Resolution
MODEL_TAG="${MODEL_TAG:-gpt-4o-mini}"
AGDIR="configs/agent_conf/RAG_Agents/${MODEL_TAG}"
AG="Structure_rag_${MODEL_TAG}-zep_512_temp0.yaml"
OUTDIR="outputs/${MODEL_TAG}-zep"
LOG_SFX=""; [ "$MODEL_TAG" != "gpt-4o-mini" ] && LOG_SFX="_${MODEL_TAG}"
rm -f "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
mkdir -p "$LOGROOT"

echo "================ Zep FC-SH ${L} (${MODEL_TAG}, chunk512 temp0) ================"; date
# N_ABLATION: optional smoke limit (e.g. N_ABLATION=2 = 2 qids only); unset = full run.
python main.py --agent_config "$AGDIR/$AG" \
  --dataset_config "$DCONF/Factconsolidation_sh_${L}.yaml" --force \
  ${N_ABLATION:+--max_test_queries_ablation "$N_ABLATION"} \
  > "$LOGROOT/run_zep_${L}${LOG_SFX}.log" 2>&1
echo "[zep ${L} ${MODEL_TAG}] exit=$?"; date
python3 -c "
import json, glob
from collections import defaultdict
fs=glob.glob('$OUTDIR/Conflict_Resolution/*sh_${L}*results*.json')
if not fs: print('no result'); raise SystemExit
gt={r['query_id']:r.get('conflict_type') for r in json.load(open('analysis/results/sh_${L}_mquake_analysis.json')) if 'query_id' in r}
rows=json.load(open(fs[0]))['data']; agg=defaultdict(lambda:[0,0])
for x in rows:
    c=gt.get(x.get('query_id'),'?'); agg[c][0]+=1; agg[c][1]+=1 if x.get('substring_exact_match') else 0
hp=agg['has_pair']; nc=agg['no_conflict_pair']; n=sum(v[0] for v in agg.values()); em=sum(v[1] for v in agg.values())
print(f'Zep FC-SH ${L}: has_pair {hp[1]}/{hp[0]} | no_conf {nc[1]}/{nc[0]} | overall {em}/{n}')
"
echo "================ DONE Zep ${L} ================"
