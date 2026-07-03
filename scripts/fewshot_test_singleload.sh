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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

RUN_ID="${RUN_ID:-main30b-test-with-gold-pubmedbert-mmr-singleload}"
RUN_DIR="exp/few_shot/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/test_singleload.log"
mkdir -p "${LOG_DIR}"

QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/public_models/bert/pubmedbert}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-hf_mean_pool}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
SHOT_POOL="${SHOT_POOL:-datasets/aligned_en_folded.csv}"
INFER_PATH="${INFER_PATH:-datasets/test_assets/test_aligned_en.csv}"
TEST_GOLD="${TEST_GOLD:-datasets/test_assets/test_gold_en.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-datasets/test_assets/test_template_en.csv}"
STRATEGY="${STRATEGY:-similarity_mmr}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
EN_ONLY="${EN_ONLY:-True}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_en.py}"

INPUT_JSON="${RUN_DIR}/input.json"
RAW_JSON="${RUN_DIR}/qwen.json"
PRED_CSV="${RUN_DIR}/prediction.csv"
AGG_JSON="${RUN_DIR}/aggregate_score.json"
SCORE_JSON="${RUN_DIR}/prediction.json"

{
  echo "[FEWSHOT TEST] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[FEWSHOT TEST] RUN_ID=${RUN_ID}"
  echo "[FEWSHOT TEST] SHOT_POOL=${SHOT_POOL}"
  echo "[FEWSHOT TEST] INFER_PATH=${INFER_PATH}"
  echo "[FEWSHOT TEST] TEST_GOLD=${TEST_GOLD}"
  echo "[FEWSHOT TEST] TEST_TEMPLATE=${TEST_TEMPLATE}"
  echo "[FEWSHOT TEST] EMBEDDING_MODEL=${EMBEDDING_MODEL}"
  echo "[FEWSHOT TEST] EMBEDDING_BACKEND=${EMBEDDING_BACKEND}"

  mkdir -p "${RUN_DIR}"
  "${PYTHON_BIN}" exp/few_shot/scripts/strategy/similarity_gold_shots.py \
    --bs_sample_from "${SHOT_POOL}" \
    --infer_path "${INFER_PATH}" \
    --output_path "${INPUT_JSON}" \
    --shot_num "${SHOT_NUM}" \
    --metrics "${METRICS}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --embedding_backend "${EMBEDDING_BACKEND}" \
    --embedding_max_length "${EMBEDDING_MAX_LENGTH}" \
    --strategy "${STRATEGY}" \
    --bootstrap_num "${BOOTSTRAP_NUM}" \
    --en_only "${EN_ONLY}"

  "${PYTHON_BIN}" model_runners/infer_qwen3.py \
    --model_path "${QWEN_MODEL}" \
    --data_path "${INPUT_JSON}" \
    --file_name "${RAW_JSON}" \
    --run_id "${RUN_ID}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  "${PYTHON_BIN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
    --prediction_path "${RAW_JSON}" \
    --true_path "${TEST_TEMPLATE}" \
    --output_csv "${PRED_CSV}" \
    --score_json "${AGG_JSON}" \
    --metrics "${METRICS}" \
    --markdown

  "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${TEST_GOLD}" "${PRED_CSV}" "${SCORE_JSON}"
  echo "[FEWSHOT TEST] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG_FILE}"
