#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_QWEN="${PYTHON_QWEN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"

DEV_RUN_ID="${DEV_RUN_ID:-dev30b-bootstrap20x7-oof-with-gold}"
DEV_NO_RUN_ID="${DEV_NO_RUN_ID:-dev30b-bootstrap20x7-oof-no-gold}"
DEV_RUN_DIR="exp/few_shot/runs/${DEV_RUN_ID}"
DEV_NO_DIR="exp/few_shot/runs/${DEV_NO_RUN_ID}"

TEST_WITH_DIR="${TEST_WITH_DIR:-exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-with-gold}"
TEST_NO_DIR="${TEST_NO_DIR:-exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-no-gold}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/en_dev_selected_bootstrap}"
DEV_X_DIR="${OUTPUT_ROOT}/dev_x_selection"
TEST_X_DIR="${OUTPUT_ROOT}/test_x_prefix_scores"
TABLE_DIR="${OUTPUT_ROOT}/main_tables"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${DEV_NO_DIR}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[DEV-SELECTED] Missing required file: ${path}" >&2
    exit 1
  fi
}

run_dev_with_gold_oof() {
  if [[ -s "${DEV_RUN_DIR}/qwen.json" ]]; then
    echo "[DEV-SELECTED] Dev with-gold raw exists; skipping OOF inference"
    return
  fi
  echo "[DEV-SELECTED] Running dev with-gold OOF bootstrap 20x7"
  PYTHON_BIN="${PYTHON_QWEN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  STRATEGY=random_bootstrap \
  SHOT_NUM="${SHOT_NUM}" \
  BOOTSTRAP_NUM="${BOOTSTRAP_NUM}" \
  RUN_ID="${DEV_RUN_ID}" \
  QWEN_MODEL="${QWEN_MODEL}" \
  METRICS="${METRICS}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  KEEP_EXISTING_INPUT=1 \
  KEEP_EXISTING_RAW=1 \
  bash scripts/fewshot_main_experiment_oof_singleload.sh
}

run_dev_no_gold_oof() {
  require_file "${DEV_RUN_DIR}/input_all_folds.json"
  local input_json="${DEV_NO_DIR}/input_all_folds.json"
  local raw_json="${DEV_NO_DIR}/qwen.json"

  if [[ ! -s "${input_json}" ]]; then
    echo "[DEV-SELECTED] Building dev no-gold OOF input from with-gold input"
    "${PYTHON_QWEN}" exp/few_shot/scripts/make_no_gold_input.py \
      --input_json "${DEV_RUN_DIR}/input_all_folds.json" \
      --output_json "${input_json}" \
      --assert_no_gold
  fi

  if [[ -s "${raw_json}" ]]; then
    echo "[DEV-SELECTED] Dev no-gold raw exists; skipping inference"
  else
    echo "[DEV-SELECTED] Running dev no-gold OOF inference"
    "${PYTHON_QWEN}" model_runners/infer_qwen3.py \
      --model_path "${QWEN_MODEL}" \
      --data_path "${input_json}" \
      --file_name "${raw_json}" \
      --run_id "${DEV_NO_RUN_ID}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  fi
}

run_prefix_sweeps() {
  echo "[DEV-SELECTED] Evaluating dev bootstrap prefixes"
  "${PYTHON_QWEN}" scripts/analysis/bootstrap_prefix_eval.py \
    --with_raw "${DEV_RUN_DIR}/qwen.json" \
    --without_raw "${DEV_NO_DIR}/qwen.json" \
    --template_csv datasets/mediqa-eval-2026-valid.csv \
    --gold_csv datasets/mediqa-eval-2026-valid_1rater_en.csv \
    --output_dir "${DEV_X_DIR}" \
    --max_x "${BOOTSTRAP_NUM}" \
    --metrics "${METRICS}" \
    --markdown

  echo "[DEV-SELECTED] Evaluating test bootstrap prefixes"
  require_file "${TEST_WITH_DIR}/qwen30b.json"
  require_file "${TEST_NO_DIR}/qwen30b.json"
  "${PYTHON_QWEN}" scripts/analysis/bootstrap_prefix_eval.py \
    --with_raw "${TEST_WITH_DIR}/qwen30b.json" \
    --without_raw "${TEST_NO_DIR}/qwen30b.json" \
    --template_csv datasets/test_assets/test_template_en.csv \
    --gold_csv datasets/test_assets/test_gold_en.csv \
    --output_dir "${TEST_X_DIR}" \
    --max_x "${BOOTSTRAP_NUM}" \
    --metrics "${METRICS}" \
    --markdown \
    --skip_selected_x
}

build_tables() {
  echo "[DEV-SELECTED] Building dev-selected English dev/test tables"
  "${PYTHON_QWEN}" scripts/analysis/build_dev_selected_english_tables.py \
    --output_dir "${TABLE_DIR}" \
    --selected_x_json "${DEV_X_DIR}/selected_x.json" \
    --dev_fewshot_with_dir "${DEV_X_DIR}/with_gold" \
    --dev_fewshot_without_dir "${DEV_X_DIR}/without_gold" \
    --test_fewshot_with_dir "${TEST_X_DIR}/with_gold" \
    --test_fewshot_without_dir "${TEST_X_DIR}/without_gold"
}

main() {
  echo "[DEV-SELECTED] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[DEV-SELECTED] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[DEV-SELECTED] QWEN_MODEL=${QWEN_MODEL}"
  echo "[DEV-SELECTED] OUTPUT_ROOT=${OUTPUT_ROOT}"
  require_file datasets/aligned_en_folded.csv
  require_file datasets/mediqa-eval-2026-valid.csv
  require_file datasets/mediqa-eval-2026-valid_1rater_en.csv
  require_file datasets/test_assets/test_template_en.csv
  require_file datasets/test_assets/test_gold_en.csv
  run_dev_with_gold_oof
  run_dev_no_gold_oof
  run_prefix_sweeps
  build_tables
  echo "[DEV-SELECTED] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee "${LOG_FILE}"
