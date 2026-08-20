#!/bin/bash
# LongMemEval-KU runner over OUR mem0 pipeline (one SHARD), then optional judge.
#   Usage: RUN_OAI_KEY_NAME=OPENAI_API_KEY_A SHARD=0 NSHARD=4 \
#            bash run_lme_ku.sh <method> [limit] [judge]
#     method = ours | b | vanilla        (zep handled separately)
#     limit  = 0 (all 78 KU) or N for smoke
#     judge  = 1 to run official evaluate_qa.py on THIS shard's hyp (default 0;
#              for sharded runs judge the concatenated hyp via the orchestrator)
#   SHARD/NSHARD (env): split the 78 KU questions across NSHARD parallel processes
#     (idx % NSHARD == SHARD). Each shard gets its OWN sub_dataset suffix -> its own
#     qdrant store + caches + hyp file, so concurrent processes never collide.
# Mirrors run_fc_sh.sh's env recipe; LongMemEval-specific isolation.
set -u
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
LME_DATA_DIR="${LME_DATA_DIR:-$REPO_ROOT/data/longmemeval}"
export LME_DATA="${LME_DATA:-$LME_DATA_DIR/longmemeval_s_ku.json}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-query_time_ku}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd $REPO_ROOT
set -a; [[ -f .env ]] && . .env; set +a

if [[ -n "${RUN_OAI_KEY_NAME:-}" ]]; then
  export OPENAI_API_KEY="${!RUN_OAI_KEY_NAME}"
  echo "[key] using \$$RUN_OAI_KEY_NAME ...${OPENAI_API_KEY: -6}"
fi
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "[key] ERROR: no OPENAI_API_KEY"; exit 1; }

METHOD="${1:?need method (ours|ours_no_p5|ours_no_p5_guard_off|ours_p5_reuse|ours_q_llm_recency|ours_q_llm_recency_2s|ours_dont_ask|b|vanilla)}"
case "$METHOD" in
  ours_struct_llm_matching) METHOD=ours_no_p5;;
  mem0_vanilla) METHOD=vanilla;;
  mem0_with_ours_extraction) METHOD=b;;
  vanilla_rag_2stage) METHOD=ours_q_llm_recency_2s;;
  dont_ask) METHOD=ours_dont_ask;;
esac
LIMIT="${2:-0}"
JUDGE="${3:-0}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
SHARDSFX=""
[[ "$NSHARD" -gt 1 ]] && SHARDSFX="_s${SHARD}n${NSHARD}"

DATA=$LME_DATA_DIR/longmemeval_s_ku.json
JUDGE_PY=$REPO_ROOT/llm_based_eval/evaluate_qa_official.py
SUBDS="longmemeval_s_ku_${METHOD}${SHARDSFX}"  # method-specific: prevents concurrent-launch race on history__<subds>__*.db glob
AGDIR=configs/agent_conf/RAG_Agents/gpt-4o-mini
LOGROOT=logs
HYPDIR=analysis/results/lme_hyps
STOREBASE=$REPO_ROOT/analysis/results/expanded/stores
PC=$PWD/analysis/results/p1_caches/lme
mkdir -p "$LOGROOT" "$PC" "$HYPDIR" "$PWD/analysis/results/phase0"

export MEM0_COST_LOG="$LOGROOT/cost_lme_${METHOD}${SHARDSFX}.jsonl"; : > "$MEM0_COST_LOG"

if [[ "$METHOD" == "ours" ]]; then
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"
  export MEM0_TRIPLE_CACHE="$PC/triple_ours${SHARDSFX}.json"
  export MEM0_SUBJECT_CACHE="$PC/subject_ours${SHARDSFX}.json"
  export MEM0_GROUPING_CACHE="$PC/grouping_ours${SHARDSFX}.json"
  export MEM0_CONFLICT_CACHE="$PC/conflict_ours${SHARDSFX}.json"
  # global (S,P) inverted index (MEM0_SP_INDEX_PATH) intentionally UNSET — ours'
  # phase2 query builds its (S,P) map on-the-fly from per-candidate triple metadata
  # and never reads the global index (only phase0_query hybrid_retrieve uses it).
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_ours${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_GROUPING_CACHE" "$MEM0_CONFLICT_CACHE" \
         "$MEM0_SUBJECT_CACHE" "$MEM0_EXTRACTION_CACHE" "$MEM0_TRIPLE_CACHE"
elif [[ "$METHOD" == "ours_no_p5" ]]; then
  # ABLATION: SAME as ours (phase2: structural + P3 LLM identity grouping),
  # BUT env MEM0_P5_SKIP=1 forces argmax on every group regardless of P5.
  # Reuses ours' P1 extraction cache so the WRITE is identical -> isolates
  # query-time P5 impact only. Aligned with FC-SH run_fc_sh.sh ours_no_p5 branch.
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_no_p5.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"  # reuse ours' (held-fixed)
  export MEM0_TRIPLE_CACHE="$PC/triple_ours_no_p5${SHARDSFX}.json"
  export MEM0_SUBJECT_CACHE="$PC/subject_ours_no_p5${SHARDSFX}.json"
  export MEM0_GROUPING_CACHE="$PC/grouping_ours_no_p5${SHARDSFX}.json"
  export MEM0_CONFLICT_CACHE="$PC/conflict_ours_no_p5${SHARDSFX}.json"    # unused with SKIP but kept for consistency
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_ours_no_p5${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  export MEM0_P5_SKIP=1
  # Do NOT rm extraction cache — held-fixed reuse from ours; rm other caches for clean run
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_GROUPING_CACHE" "$MEM0_CONFLICT_CACHE" \
         "$MEM0_SUBJECT_CACHE" "$MEM0_TRIPLE_CACHE"
elif [[ "$METHOD" == "ours_p5_reuse" ]]; then
  # REUSE ours_no_p5's populated store + write-time caches, only re-run query phase
  # with P5 enabled. Isolates P5 decision trace WITHOUT rebuilding memory pool.
  # Uses ours_no_p5.yaml (same qdrant store path + collection_name as ours_no_p5 run)
  # and UNSETS MEM0_P5_SKIP so P5 activates at query-time.
  # Requires: ours_no_p5 s0/s1 must have completed with full 78 KU coverage.
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_no_p5.yaml  # matches ours_no_p5 store path
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"       # reuse (unchanged)
  export MEM0_TRIPLE_CACHE="$PC/triple_ours_no_p5${SHARDSFX}.json"          # reuse ours_no_p5's
  export MEM0_SUBJECT_CACHE="$PC/subject_ours_no_p5${SHARDSFX}.json"        # reuse ours_no_p5's
  export MEM0_GROUPING_CACHE="$PC/grouping_ours_no_p5${SHARDSFX}.json"      # reuse ours_no_p5's
  export MEM0_CONFLICT_CACHE="$PC/conflict_ours_p5_reuse${SHARDSFX}.json"   # FRESH — P5 decisions to be logged
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_ours_p5_reuse${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  unset MEM0_P5_SKIP   # P5 enabled at query time (critical — this is the only diff vs ours_no_p5)
  # Override SUBDS to reuse ours_no_p5's store namespace (both store path + user_id namespace).
  # ours_no_p5 uses SUBDS="longmemeval_s_ku_ours_no_p5${SHARDSFX}" via method-specific fix
  SUBDS="longmemeval_s_ku_ours_no_p5${SHARDSFX}"
  # Only rm the FRESH state (conflict cache + cand log); NEVER rm the reused caches/store
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_CONFLICT_CACHE"
  # QUERY_ONLY flag passed to python (skip memorize step)
  QUERY_ONLY_FLAG="--query-only"
elif [[ "$METHOD" == "ours_no_p5_guard_off" ]]; then
  # Subject-consistency guard ablation on LME (2026-07-17). Mirror
  # ours_p5_reuse pattern exactly (reuse ours_no_p5 store + write caches +
  # query-only) but instead of enabling P5 we set MEM0_SUBJECT_GUARD_OFF=1
  # so llm_identity_clusters accepts every LLM-proposed cluster (no
  # cross-subject rejection). Isolates the guard's contribution to LME EM.
  # Requires: ours_no_p5 s0/s1 must have completed with full 78 KU coverage.
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_no_p5_guard_off.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"       # reuse (unchanged)
  export MEM0_TRIPLE_CACHE="$PC/triple_ours_no_p5${SHARDSFX}.json"          # reuse ours_no_p5's
  export MEM0_SUBJECT_CACHE="$PC/subject_ours_no_p5${SHARDSFX}.json"        # reuse ours_no_p5's
  export MEM0_GROUPING_CACHE="$PC/grouping_ours_no_p5${SHARDSFX}.json"      # reuse ours_no_p5's (LLM raw output cached; guard applied at use-time)
  export MEM0_CONFLICT_CACHE="$PC/conflict_ours_no_p5_guard_off${SHARDSFX}.json"  # unused with P5 SKIP but fresh
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_ours_no_p5_guard_off${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  export MEM0_P5_SKIP=1
  export MEM0_SUBJECT_GUARD_OFF=1                                    # <-- the ONE thing that differs from ours_no_p5
  SUBDS="longmemeval_s_ku_ours_no_p5${SHARDSFX}"                     # reuse ours_no_p5's namespace
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_CONFLICT_CACHE"
  QUERY_ONLY_FLAG="--query-only"
elif [[ "$METHOD" == "ours_q_llm_recency" ]]; then
  # Q-llm-recency baseline (2026-07-11) on LME-KU: naive fact-level RAG + LLM
  # does recency judgment. Reuses ours_no_p5's populated store + write caches
  # byte-for-byte (like ours_p5_reuse pattern) and only re-runs query phase
  # with MEM0_QUERY_MODE=q_llm_recency + MEM0_P5_SKIP=1. Directly tests C1
  # arm ("LLM 全程不參與 recency 裁決") on natural-language / KU chain data.
  # Uses factconsolidation.rag_agent template (via MEM0_Q_LLM_RECENCY_TEMPLATE_DS
  # override) so cross-dataset the LLM gets the same recency-rule instruction.
  # Requires: ours_no_p5 must have completed with full 78 KU coverage on the
  # same SHARD/NSHARD split.
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_q_llm_recency.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"       # reuse (unchanged)
  export MEM0_TRIPLE_CACHE="$PC/triple_ours_no_p5${SHARDSFX}.json"          # reuse ours_no_p5's
  export MEM0_SUBJECT_CACHE="$PC/subject_ours_no_p5${SHARDSFX}.json"        # reuse ours_no_p5's
  export MEM0_GROUPING_CACHE="$PC/grouping_ours_no_p5${SHARDSFX}.json"      # reuse ours_no_p5's
  export MEM0_CONFLICT_CACHE="$PC/conflict_q_llm_recency${SHARDSFX}.json"   # unused with SKIP but fresh
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_q_llm_recency${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=q_llm_recency
  export MEM0_P5_SKIP=1
  export MEM0_Q_LLM_RECENCY_TOPK="${MEM0_Q_LLM_RECENCY_TOPK:-100}"          # canonical top-100
  export MEM0_Q_LLM_RECENCY_TEMPLATE_DS=factconsolidation_sh                 # cross-dataset canonical recency prompt (must contain "factconsolidation_" for DATASET_MAPPING prefix match)
  # Reuse ours_no_p5's store + user_id namespace
  SUBDS="longmemeval_s_ku_ours_no_p5${SHARDSFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_CONFLICT_CACHE"
  QUERY_ONLY_FLAG="--query-only"
elif [[ "$METHOD" == "ours_q_llm_recency_2s" ]]; then
  # Two-stage Q-llm-recency on LME-KU (2026-07-14): removes the FC-SH template
  # override that made single-stage ours_q_llm_recency an unfair prompt-augmented
  # baseline on LME (LME native rag_agent lacks recency instruction). Stage 1
  # uses FC-SH `factconsolidation.rag_agent`-mirror prompt to pick winner
  # serials (hardcoded in agent.py:q_llm_recency_two_stage branch); Stage 2
  # uses LME native rag_agent (byte-identical to ours main / vanilla / b).
  # Reuses ours_no_p5's populated store + write caches (query-only).
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_q_llm_recency.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"       # reuse (unchanged)
  export MEM0_TRIPLE_CACHE="$PC/triple_ours_no_p5${SHARDSFX}.json"          # reuse ours_no_p5's
  export MEM0_SUBJECT_CACHE="$PC/subject_ours_no_p5${SHARDSFX}.json"        # reuse ours_no_p5's
  export MEM0_GROUPING_CACHE="$PC/grouping_ours_no_p5${SHARDSFX}.json"      # reuse ours_no_p5's
  export MEM0_CONFLICT_CACHE="$PC/conflict_q_llm_recency_2s${SHARDSFX}.json" # unused with SKIP
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_q_llm_recency_2s${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=q_llm_recency_two_stage
  export MEM0_P5_SKIP=1
  export MEM0_Q_LLM_RECENCY_TOPK="${MEM0_Q_LLM_RECENCY_TOPK:-100}"          # canonical top-100
  # NOTE: intentionally NOT setting MEM0_Q_LLM_RECENCY_TEMPLATE_DS. Stage 2
  # uses LME native rag_agent template (else-branch in agent.py), same as
  # ours main / vanilla / b — this is the fairness fix vs single-stage.
  # Reuse ours_no_p5's store + user_id namespace (same as single-stage).
  SUBDS="longmemeval_s_ku_ours_no_p5${SHARDSFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_CONFLICT_CACHE"
  QUERY_ONLY_FLAG="--query-only"
elif [[ "$METHOD" == "ours_dont_ask" ]]; then
  # Don't Ask (Reddy & Challaram, 2026) faithful port to LME-KU (2026-07-20).
  # Reference mechanism: single LLM candidate-extraction over ordinal-numbered
  # top-100 pool -> deterministic max(serial) -> output that candidate's
  # answer_entity (NO answer-generation step). Implemented as
  # MEM0_QUERY_MODE=dont_ask in agent.py. Reuses ours_no_p5's populated store +
  # write caches byte-for-byte (query-only, same as q_llm_recency pattern) so
  # the bank + per-user_id `ordinal` recency signal are IDENTICAL to ours; only
  # the query-time KU mechanism differs (LLM candidate extraction in place of
  # ours' (S,P) structural grouping). Requires: ours_no_p5 must have completed
  # with full 78 KU coverage on the same SHARD/NSHARD split.
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_q_llm_recency.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"       # reuse (unchanged)
  export MEM0_TRIPLE_CACHE="$PC/triple_ours_no_p5${SHARDSFX}.json"          # reuse ours_no_p5's
  export MEM0_SUBJECT_CACHE="$PC/subject_ours_no_p5${SHARDSFX}.json"        # reuse ours_no_p5's
  export MEM0_GROUPING_CACHE="$PC/grouping_ours_no_p5${SHARDSFX}.json"      # reuse ours_no_p5's
  export MEM0_CONFLICT_CACHE="$PC/conflict_dont_ask${SHARDSFX}.json"        # unused by dont_ask but fresh
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_dont_ask${SHARDSFX}_p1"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=dont_ask
  export MEM0_P5_SKIP=1
  export MEM0_DONT_ASK_TOPK="${MEM0_DONT_ASK_TOPK:-100}"                    # canonical top-100
  # Reuse ours_no_p5's store + user_id namespace
  SUBDS="longmemeval_s_ku_ours_no_p5${SHARDSFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_CONFLICT_CACHE"
  QUERY_ONLY_FLAG="--query-only"
elif [[ "$METHOD" == "b" ]]; then
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_unified_dest.yaml
  export MEM0_TRIPLE_MODEL=gpt-4o-mini
  export MEM0_EXTRACTION_CACHE="$PC/extraction_ours${SHARDSFX}.json"  # held-fixed P1 (reuse ours' shard cache)
  unset MEM0_ADD_MODE MEM0_QUERY_MODE MEM0_TRIPLE_CACHE
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_b${SHARDSFX}_p1"
  rm -rf "$MEM0_CAND_LOG_DIR"
else  # vanilla
  AG=Structure_rag_gpt-4o-mini-mem0_512_openai_native.yaml
  unset MEM0_ADD_MODE MEM0_QUERY_MODE MEM0_EXTRACTION_CACHE MEM0_TRIPLE_CACHE
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/lme_vanilla${SHARDSFX}_p1"
  rm -rf "$MEM0_CAND_LOG_DIR"
fi
mkdir -p "$MEM0_CAND_LOG_DIR"

# fresh store for this method+shard. agent.py:273 suffixes the yaml `path:` field
# (NOT collection_name) with __<sub_dataset>, and isolates history.db per
# <sub_dataset>__<agent_fp>. Clean both so a re-run starts empty.
# EXCEPTION: ours_p5_reuse mode reuses ours_no_p5's populated store — do NOT rm.
BASEPATH=$(grep -oP '^\s*path:\s*\K\S+' "$AGDIR/$AG" | head -1)
if [[ "$METHOD" != "ours_p5_reuse" ]]; then
  rm -rf "${BASEPATH}__${SUBDS}" 2>/dev/null
  rm -f "$HOME/.mem0/history__${SUBDS}__"*.db 2>/dev/null
fi

HYP="$HYPDIR/lme_ku_${METHOD}${SHARDSFX}.jsonl"
[[ "$LIMIT" != "0" ]] && HYP="$HYPDIR/lme_ku_${METHOD}_smoke${LIMIT}.jsonl"
rm -f "$HYP"
HYP_ABS="$PWD/$HYP"

echo "================ LME-KU $METHOD shard $SHARD/$NSHARD (limit=$LIMIT) ================"; date
python scripts/run_longmemeval_ku.py \
  --agent_config "$AGDIR/$AG" --data "$DATA" --out "$HYP" \
  --sub_dataset "$SUBDS" --qtype knowledge-update --limit "$LIMIT" \
  --shard "$SHARD" --nshard "$NSHARD" ${QUERY_ONLY_FLAG:-}
RC=$?
date; echo "[lme $METHOD shard $SHARD/$NSHARD] gen exit=$RC"
[[ $RC -ne 0 ]] && exit $RC

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"   # cheap for validation; gpt-4o for paper-final
if [[ "$JUDGE" == "1" ]]; then
  echo "---- official judge ($JUDGE_MODEL; paper-final=gpt-4o) ----"
  ( cd "$(dirname "$JUDGE_PY")" && OPENAI_API_KEY="$OPENAI_API_KEY" python "$(basename "$JUDGE_PY")" "$JUDGE_MODEL" "$HYP_ABS" "$DATA" ) \
    2>&1 || echo "[judge] failed (deps? run separately)"
fi
echo "================ DONE LME-KU $METHOD shard $SHARD/$NSHARD ================"
