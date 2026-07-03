#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/public_models/bge-m3}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-huggingface}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
RERANK_MODEL="${RERANK_MODEL:-/data/public_models/bge-reranker-large}"
DISABLE_RERANK="${DISABLE_RERANK:-0}"
RAG_PERSIST_DIR="${RAG_PERSIST_DIR:-RAG/storage}"
REBUILD_RAG_INDEX="${REBUILD_RAG_INDEX:-0}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
VALID_PATH="${VALID_PATH:-datasets/mediqa-eval-2026-valid_1rater_en.csv}"
TRUE_CSV="${TRUE_CSV:-datasets/mediqa-eval-2026-valid.csv}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SAMPLE_N_PER_FOLD="${SAMPLE_N_PER_FOLD:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_tables_30b_shot20}"
mkdir -p "${OUTPUT_ROOT}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/run_main_tables_30b_shot20_$(date +%Y%m%d_%H%M%S).log"

WITH_GOLD_RUN_ID="${WITH_GOLD_RUN_ID:-main30b-shot20-with-gold-similarity_mmr-oof-singleload}"
WITH_GOLD_RUN_DIR="exp/few_shot/runs/${WITH_GOLD_RUN_ID}"
NO_GOLD_RUN_ID="${NO_GOLD_RUN_ID:-main30b-shot20-no-gold-similarity_mmr-oof-singleload}"
NO_GOLD_RUN_DIR="exp/few_shot/runs/${NO_GOLD_RUN_ID}"
RAG_RUN_ID="${RAG_RUN_ID:-main30b-shot20-no-gold-rag-v2-query-similarity_mmr-oof-singleload}"
RAG_RUN_DIR="exp/few_shot/runs/${RAG_RUN_ID}"

run_qwen_json() {
  local run_id="$1"
  local run_dir="$2"
  local input_json="$3"
  local raw_json="${run_dir}/qwen.json"
  local pred_csv="${run_dir}/prediction.csv"
  local agg_score="${run_dir}/aggregate_score.json"
  local score_json="${run_dir}/prediction.json"

  mkdir -p "${run_dir}/logs"
  "${PYTHON_BIN}" model_runners/infer_qwen3.py \
    --model_path "${QWEN_MODEL}" \
    --data_path "${input_json}" \
    --file_name "${raw_json}" \
    --run_id "${run_id}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  "${PYTHON_BIN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
    --prediction_path "${raw_json}" \
    --true_path "${TRUE_CSV}" \
    --output_csv "${pred_csv}" \
    --score_json "${agg_score}" \
    --metrics "${METRICS}" \
    --markdown

  "${PYTHON_BIN}" eval_en.py "${VALID_PATH}" "${pred_csv}" "${score_json}"
}

run_with_gold() {
  QWEN_MODEL="${QWEN_MODEL}" \
  EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
  EMBEDDING_BACKEND="${EMBEDDING_BACKEND}" \
  EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH}" \
  STRATEGY="similarity_mmr" \
  SHOT_NUM="20" \
  BOOTSTRAP_NUM="1" \
  RUN_ID="${WITH_GOLD_RUN_ID}" \
  KEEP_EXISTING_INPUT=0 \
  KEEP_EXISTING_RAW=0 \
  SAMPLE_N_PER_FOLD="${SAMPLE_N_PER_FOLD}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  bash scripts/fewshot_main_experiment_oof_singleload.sh
}

prepare_no_gold_input() {
  mkdir -p "${NO_GOLD_RUN_DIR}"
  cp -f "${WITH_GOLD_RUN_DIR}/input_all_folds.json" "${NO_GOLD_RUN_DIR}/input_all_folds.with_gold_source.json"
  "${PYTHON_BIN}" exp/few_shot/scripts/make_no_gold_input.py \
    --input_json "${WITH_GOLD_RUN_DIR}/input_all_folds.json" \
    --output_json "${NO_GOLD_RUN_DIR}/input_all_folds.json" \
    --assert_no_gold
}

run_no_gold() {
  prepare_no_gold_input
  run_qwen_json "${NO_GOLD_RUN_ID}" "${NO_GOLD_RUN_DIR}" "${NO_GOLD_RUN_DIR}/input_all_folds.json"
}

run_rag() {
  mkdir -p "${RAG_RUN_DIR}"
  if [[ "${REBUILD_RAG_INDEX}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/rag_external/build_rag_index.py \
      --persist_dir "${RAG_PERSIST_DIR}" \
      --embed_model "${EMBEDDING_MODEL}" \
      --embedding_backend "${EMBEDDING_BACKEND}" \
      --embedding_max_length "${EMBEDDING_MAX_LENGTH}"
  fi

  local rerank_args=()
  if [[ "${DISABLE_RERANK}" == "1" ]]; then
    rerank_args+=(--disable_rerank)
  fi

  "${PYTHON_BIN}" scripts/rag_external/augment_rag_input.py \
    --input_json "${NO_GOLD_RUN_DIR}/input_all_folds.json" \
    --output_json "${RAG_RUN_DIR}/input_all_folds.json" \
    --persist_dir "${RAG_PERSIST_DIR}" \
    --embed_model "${EMBEDDING_MODEL}" \
    --embedding_backend "${EMBEDDING_BACKEND}" \
    --embedding_max_length "${EMBEDDING_MAX_LENGTH}" \
    --rerank_model "${RERANK_MODEL}" \
    --query_source target_query \
    --insert_location after_llm_response \
    --retrieve_top_k 10 \
    --rerank_top_n 2 \
    --max_evidence_chars 1200 \
    --include_metadata \
    "${rerank_args[@]}"
  run_qwen_json "${RAG_RUN_ID}" "${RAG_RUN_DIR}" "${RAG_RUN_DIR}/input_all_folds.json"
}

main() {
  echo "[RAG DEV] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[RAG DEV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[RAG DEV] QWEN_MODEL=${QWEN_MODEL}"
  echo "[RAG DEV] EMBEDDING_MODEL=${EMBEDDING_MODEL}"
  echo "[RAG DEV] EMBEDDING_BACKEND=${EMBEDDING_BACKEND}"
  echo "[RAG DEV] RAG_PERSIST_DIR=${RAG_PERSIST_DIR}"
  echo "[RAG DEV] DISABLE_RERANK=${DISABLE_RERANK}"
  run_with_gold
  run_no_gold
  run_rag
  echo "[RAG DEV] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee "${LOG_FILE}"
