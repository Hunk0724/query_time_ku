#!/bin/bash
# Parametrized FC-SH runner (one script for all lengths/methods; parallel-safe).
#   Usage:  RUN_OAI_KEY="$OPENAI_API_KEY_B" bash run_fc_sh.sh <L> <method>
#     L      = 6k | 32k | 64k | 262k
#     method = ours | vanilla
#   - ours    : P1 unified extractor + phase0 conservative write + phase2 query
#               resolution + raw-q retrieval + batch embedding (the defined method)
#   - vanilla : stock mem0 (native extraction, destructive update) + raw-q (ungated)
# Per-run isolated store/caches/cost-log keyed by <method>_<L> -> safe to run many
# in parallel, each with its own OpenAI key (RUN_OAI_KEY) so rate limits don't collide.
set -u
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
LME_DATA_DIR="${LME_DATA_DIR:-$REPO_ROOT/data/longmemeval}"
export LME_DATA="${LME_DATA:-$LME_DATA_DIR/longmemeval_s_cleaned.json}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-repro}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd $REPO_ROOT
set -a; [[ -f .env ]] && . .env; set +a

# per-run API key (override so parallel runs don't share quota). Two ways:
#   RUN_OAI_KEY_NAME=OPENAI_API_KEY_A  -> resolved from .env here (robust; preferred)
#   RUN_OAI_KEY=sk-...                 -> literal key
if [[ -n "${RUN_OAI_KEY_NAME:-}" ]]; then
  export OPENAI_API_KEY="${!RUN_OAI_KEY_NAME}"
  echo "[key] using \$$RUN_OAI_KEY_NAME ...${OPENAI_API_KEY: -6}"
elif [[ -n "${RUN_OAI_KEY:-}" ]]; then
  export OPENAI_API_KEY="$RUN_OAI_KEY"
  echo "[key] using literal key ...${RUN_OAI_KEY: -6}"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then echo "[key] ERROR: no OPENAI_API_KEY resolved"; exit 1; fi

L="${1:?need L (6k|32k|64k|262k)}"
METHOD="${2:?need method (ours|ours_struct|ours_no_p5|ours_no_p5_guard_off|ours_p3_only_no_struct|ours_q_llm_recency|b|vanilla)}"
case "$METHOD" in
  ours_struct_llm_matching) METHOD=ours_no_p5;;
  ours_struct_only) METHOD=ours_struct;;
  ours_llm_matching_only) METHOD=ours_p3_only_no_struct;;
  mem0_vanilla) METHOD=vanilla;;
  mem0_with_ours_extraction) METHOD=b;;
  vanilla_rag) METHOD=ours_q_llm_recency;;
esac
LOGROOT=logs
DCONF=configs/data_conf/Conflict_Resolution

# MODEL_TAG: empty = default gpt-4o-mini (back-compat); non-empty = isolate store/
# cache/output/log paths per model so cross-model runs do not collide. yaml file
# AGDIR auto-switches when MODEL_TAG is set (expects configs/agent_conf/RAG_Agents/
# ${MODEL_TAG}/ to exist).
MODEL_TAG="${MODEL_TAG:-}"
TAG_SFX=""; [ -n "$MODEL_TAG" ] && TAG_SFX="__${MODEL_TAG}"
if [ -n "$MODEL_TAG" ]; then
  AGDIR="configs/agent_conf/RAG_Agents/${MODEL_TAG}"
else
  AGDIR="configs/agent_conf/RAG_Agents/gpt-4o-mini"
fi
STOREBASE=$REPO_ROOT/analysis/results/expanded/stores
PC="$PWD/analysis/results/p1_caches${TAG_SFX}"
mkdir -p "$LOGROOT" "$PC" "$PWD/analysis/results/phase0"

# cost/latency log (env-gated instrumentation; fresh per run)
export MEM0_COST_LOG="$LOGROOT/cost_${METHOD}_${L}${TAG_SFX}.jsonl"
: > "$MEM0_COST_LOG"

if [[ "$METHOD" == "ours" ]]; then
  TAG=unified
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified.yaml"
  STORE="qdrant_gpt4o_512_openai_unified${TAG_SFX}__factconsolidation_sh_${L}"
  export MEM0_TRIPLE_MODEL="${MEM0_TRIPLE_MODEL:-gpt-4o-mini}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"
  export MEM0_TRIPLE_CACHE="$PC/triple_cache_p1_${L}.json"
  export MEM0_SUBJECT_CACHE="$PC/subject_cache_p1_${L}.json"
  export MEM0_GROUPING_CACHE="$PC/grouping_cache_p1_${L}.json"
  export MEM0_CONFLICT_CACHE="$PC/conflict_cache_p1_${L}.json"
  export MEM0_SP_INDEX_PATH="$PWD/analysis/results/phase0/sp_index_p1_sh_${L}${TAG_SFX}.json"
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_p1${TAG_SFX}"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified${TAG_SFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_SP_INDEX_PATH" "$MEM0_GROUPING_CACHE" \
         "$MEM0_CONFLICT_CACHE" "$STOREBASE/$STORE" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
elif [[ "$METHOD" == "ours_struct" ]]; then
  # ABLATION (next-step #1): SAME conservative write as ours, but query-time =
  # STRUCTURAL only ((S,P) group + deterministic temporal; NO LLM grouping, NO
  # conflict-type). Reuses ours' P1 extraction caches so the WRITE is identical ->
  # isolates how much of ours' has_pair EM comes purely from structural+temporal.
  # Compare against `ours` (phase2) at the same length.
  TAG=unified_struct
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_struct.yaml"
  STORE="qdrant_gpt4o_512_openai_unified_struct${TAG_SFX}__factconsolidation_sh_${L}"
  export MEM0_TRIPLE_MODEL="${MEM0_TRIPLE_MODEL:-gpt-4o-mini}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"   # reuse ours' (held-fixed write)
  export MEM0_TRIPLE_CACHE="$PC/triple_cache_p1_${L}.json"
  export MEM0_SUBJECT_CACHE="$PC/subject_cache_p1_${L}.json"
  export MEM0_GROUPING_CACHE="$PC/grouping_cache_struct_${L}.json"
  export MEM0_CONFLICT_CACHE="$PC/conflict_cache_struct_${L}.json"
  export MEM0_SP_INDEX_PATH="$PWD/analysis/results/phase0/sp_index_struct_sh_${L}${TAG_SFX}.json"
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_struct${TAG_SFX}"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=structural
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_struct${TAG_SFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_SP_INDEX_PATH" "$STOREBASE/$STORE" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_struct${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
elif [[ "$METHOD" == "ours_p3_only_no_struct" ]]; then
  # ABLATION: apple-to-apple compare with ours_struct. Both use argmax on every
  # group; ours_struct groups by (S,P), this mode groups by P3 LLM identity
  # over the full top-k (no (S,P) partitioning). Reveals whether structural
  # (S,P) or LLM semantic identity is the stronger identity resolver.
  # Reuses ours' P1 extraction caches so the WRITE is identical.
  TAG=unified_p3_only
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_p3_only_no_struct.yaml"
  STORE="qdrant_gpt4o_512_openai_unified_p3_only_no_struct${TAG_SFX}__factconsolidation_sh_${L}"
  export MEM0_TRIPLE_MODEL="${MEM0_TRIPLE_MODEL:-gpt-4o-mini}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"   # reuse ours' (held-fixed write)
  export MEM0_TRIPLE_CACHE="$PC/triple_cache_p1_${L}.json"
  export MEM0_SUBJECT_CACHE="$PC/subject_cache_p1_${L}.json"
  export MEM0_GROUPING_CACHE="$PC/grouping_cache_p3_only_${L}.json"
  export MEM0_CONFLICT_CACHE="$PC/conflict_cache_p3_only_${L}.json"  # unused with P5_SKIP but path kept
  export MEM0_SP_INDEX_PATH="$PWD/analysis/results/phase0/sp_index_p3_only_sh_${L}${TAG_SFX}.json"
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_p3_only${TAG_SFX}"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  export MEM0_P5_SKIP=1
  export MEM0_STRUCTURAL_SKIP=1
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_p3_only_no_struct${TAG_SFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_SP_INDEX_PATH" "$MEM0_GROUPING_CACHE" \
         "$MEM0_CONFLICT_CACHE" "$STOREBASE/$STORE" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_p3_only_no_struct${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
elif [[ "$METHOD" == "ours_no_p5" ]]; then
  # ABLATION: SAME as ours (phase2: structural + P3 LLM identity grouping),
  # BUT env MEM0_P5_SKIP=1 forces argmax on every group regardless of P5
  # conflict-type. Isolates P3 upside from P5 net effect in FC world-fact
  # + counter-factual scenarios (where ~all real conflicts are FRESHNESS).
  # Reuses ours' P1 extraction caches so the WRITE is identical -> isolates
  # query-time P5 impact only.
  TAG=unified_no_p5
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_no_p5.yaml"
  STORE="qdrant_gpt4o_512_openai_unified_no_p5${TAG_SFX}__factconsolidation_sh_${L}"
  export MEM0_TRIPLE_MODEL="${MEM0_TRIPLE_MODEL:-gpt-4o-mini}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"   # reuse ours' (held-fixed write)
  export MEM0_TRIPLE_CACHE="$PC/triple_cache_p1_${L}.json"
  export MEM0_SUBJECT_CACHE="$PC/subject_cache_p1_${L}.json"
  export MEM0_GROUPING_CACHE="$PC/grouping_cache_no_p5_${L}.json"
  export MEM0_CONFLICT_CACHE="$PC/conflict_cache_no_p5_${L}.json"    # unused with SKIP but kept for consistency
  export MEM0_SP_INDEX_PATH="$PWD/analysis/results/phase0/sp_index_no_p5_sh_${L}${TAG_SFX}.json"
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_no_p5${TAG_SFX}"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  export MEM0_P5_SKIP=1
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_no_p5${TAG_SFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$MEM0_SP_INDEX_PATH" "$MEM0_GROUPING_CACHE" \
         "$MEM0_CONFLICT_CACHE" "$STOREBASE/$STORE" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_no_p5${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
elif [[ "$METHOD" == "ours_no_p5_guard_off" ]]; then
  # Subject-consistency guard ablation (2026-07-17): mirror ours_no_p5 exactly
  # BUT set MEM0_SUBJECT_GUARD_OFF=1 so llm_identity_clusters accepts every
  # LLM-proposed cluster (no cross-subject rejection). Reuses the ours_no_p5
  # populated store + all write-time caches byte-for-byte; only query behavior
  # differs (guard bypassed). Isolates the guard's contribution to final EM.
  # Requires: ours_no_p5 must have been run first at this length.
  TAG=unified_no_p5_guard_off
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_no_p5_guard_off.yaml"
  STORE="qdrant_gpt4o_512_openai_unified_no_p5${TAG_SFX}__factconsolidation_sh_${L}"   # shared with ours_no_p5
  export MEM0_TRIPLE_MODEL="${MEM0_TRIPLE_MODEL:-gpt-4o-mini}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"    # reuse ours_no_p5
  export MEM0_TRIPLE_CACHE="$PC/triple_cache_p1_${L}.json"            # reuse ours_no_p5
  export MEM0_SUBJECT_CACHE="$PC/subject_cache_p1_${L}.json"          # reuse ours_no_p5
  export MEM0_GROUPING_CACHE="$PC/grouping_cache_no_p5_${L}.json"     # reuse ours_no_p5 (LLM raw output cached; guard applied at use-time)
  export MEM0_CONFLICT_CACHE="$PC/conflict_cache_no_p5_${L}.json"     # reuse
  export MEM0_SP_INDEX_PATH="$PWD/analysis/results/phase0/sp_index_no_p5_sh_${L}${TAG_SFX}.json"
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_no_p5_guard_off${TAG_SFX}"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=phase2
  export MEM0_P5_SKIP=1
  export MEM0_SUBJECT_GUARD_OFF=1                                    # <-- the ONE thing that differs from ours_no_p5
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_no_p5_guard_off${TAG_SFX}"
  # DO NOT rm the shared store / caches (would destroy ours_no_p5). Only clean
  # this run's fresh outputs.
  rm -rf "$MEM0_CAND_LOG_DIR" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_no_p5_guard_off${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
elif [[ "$METHOD" == "ours_q_llm_recency" ]]; then
  # Q-llm-recency baseline: naive fact-level RAG + LLM does recency judgment.
  # Uses a dedicated yaml (unified_q_llm_recency.yaml) that shares the qdrant
  # store path/collection with unified_no_p5 → same populated store byte-for-byte,
  # but distinct agent_name + output_dir so ours_no_p5's results don't get
  # overwritten. See yaml header for details.
  # Requires ours_no_p5 to have been run first at this length (writes to the
  # shared store there).
  # See experiment.md §M-3 / §M-4 and experiment_prove_main_claim.md §4.8.2 #2.
  TAG=unified_q_llm_recency
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_q_llm_recency.yaml"
  STORE="qdrant_gpt4o_512_openai_unified_no_p5${TAG_SFX}__factconsolidation_sh_${L}"   # shared with ours_no_p5
  export MEM0_TRIPLE_MODEL="${MEM0_TRIPLE_MODEL:-gpt-4o-mini}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"    # reuse ours_no_p5
  export MEM0_TRIPLE_CACHE="$PC/triple_cache_p1_${L}.json"            # reuse ours_no_p5
  export MEM0_SUBJECT_CACHE="$PC/subject_cache_p1_${L}.json"          # reuse ours_no_p5
  export MEM0_GROUPING_CACHE="$PC/grouping_cache_no_p5_${L}.json"     # reuse ours_no_p5 (unused at query)
  export MEM0_CONFLICT_CACHE="$PC/conflict_cache_no_p5_${L}.json"     # reuse
  export MEM0_SP_INDEX_PATH="$PWD/analysis/results/phase0/sp_index_no_p5_sh_${L}${TAG_SFX}.json"
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_q_llm_recency${TAG_SFX}"
  export MEM0_ADD_MODE=phase0_structural
  export MEM0_QUERY_MODE=q_llm_recency
  export MEM0_P5_SKIP=1                    # bypass phase2 resolve (q_llm_recency branch replaces it)
  export MEM0_Q_LLM_RECENCY_TOPK="${MEM0_Q_LLM_RECENCY_TOPK:-100}"  # canonical: top-100 to match ours main / (b) / vanilla top-K (fair comparison; §M-4). Override to 10 for sensitivity.
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_q_llm_recency${TAG_SFX}"
  # DO NOT rm the shared store / caches (would destroy ours_no_p5). Only clean
  # this run's fresh outputs (cand log dir + prior stale q_llm_recency results).
  rm -rf "$MEM0_CAND_LOG_DIR" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_q_llm_recency${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
elif [[ "$METHOD" == "b" ]]; then
  # mem0(b): P1 extraction HELD FIXED (reuse ours' p1 extraction cache, read-only)
  # + mem0 DESTRUCTIVE update (no phase env). Isolates write-time-update loss.
  TAG=unified_dest
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_unified_dest.yaml"
  STORE="qdrant_gpt4o_512_openai_unified_dest${TAG_SFX}__factconsolidation_sh_${L}"
  export MEM0_EXTRACTION_CACHE="$PC/extraction_cache_p1_${L}.json"   # held-fixed P1
  unset MEM0_ADD_MODE MEM0_QUERY_MODE MEM0_TRIPLE_CACHE             # destructive update
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_b${TAG_SFX}"
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-unified_dest${TAG_SFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$STOREBASE/$STORE" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_unified_dest${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
else
  TAG=native
  AG="Structure_rag_${MODEL_TAG:-gpt-4o-mini}-mem0_512_openai_native.yaml"
  STORE="qdrant_gpt4o_512_openai_native${TAG_SFX}__factconsolidation_sh_${L}"
  # vanilla: native extraction + destructive update; NO phase env, NO p1 caches.
  unset MEM0_ADD_MODE MEM0_QUERY_MODE MEM0_EXTRACTION_CACHE MEM0_TRIPLE_CACHE
  export MEM0_CAND_LOG_DIR="$PWD/$LOGROOT/sh_${L}_native${TAG_SFX}"
  OUTDIR="outputs/gpt-4o-mini-mem0-chunk512-temp0-openai-native${TAG_SFX}"
  rm -rf "$MEM0_CAND_LOG_DIR" "$STOREBASE/$STORE" \
         "$OUTDIR/Conflict_Resolution/"*sh_${L}*results*.json
  ANSDIR="outputs/rag_retrieved/Structure_rag_gpt-4o-mini-mem0_512_openai_native${TAG_SFX}/k_100/factconsolidation_sh_${L}/chunksize_512"
fi
mkdir -p "$MEM0_CAND_LOG_DIR"

echo "================ $METHOD FC-SH ${L} ================"; date
# N_ABLATION: optional smoke limit (e.g. N_ABLATION=2 = only 2 qids per context).
# Full run: leave N_ABLATION unset.
python main.py --agent_config "$AGDIR/$AG" \
  --dataset_config "$DCONF/Factconsolidation_sh_${L}.yaml" --force \
  ${N_ABLATION:+--max_test_queries_ablation "$N_ABLATION"} \
  > "$LOGROOT/run_${METHOD}_${L}.log" 2>&1
echo "[$METHOD ${L}] exit=$?"; date

echo "---- EM ----"
python3 -c "
import json, glob
fs = glob.glob('$OUTDIR/**/*sh_${L}*results*.json', recursive=True)
if fs:
    rows = json.load(open(fs[0]))['data']; em = sum(1 for x in rows if x.get('substring_exact_match'))
    print(f'$METHOD FC-SH ${L}: sEM {em}/{len(rows)} = {em/len(rows)*100:.1f}%  (the paper-table metric)')
"
echo "---- cost ----"
python3 scripts/summarize_cost.py "$MEM0_COST_LOG" "$ANSDIR"
echo "================ DONE $METHOD ${L} ================"
