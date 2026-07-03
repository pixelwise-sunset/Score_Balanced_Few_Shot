#!/usr/bin/env bash
set -euo pipefail

BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
REPRO_REPO="${REPRO_REPO:-/workspace/MEDIQA-SUATBMI-ENSEMBLE-repro}"
WRAPPER_DIR="${WRAPPER_DIR:-${BASE_PROJECT}/scripts/paper_repro}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
OUT_ROOT="${OUT_ROOT:-${BASE_PROJECT}/results/zh_fewshot_bootstrap20x7_original_20260618}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['factual-consistency-wgold','writing-style']}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPRO_REPO}:${BASE_PROJECT}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}/logs"
LOG_FILE="${OUT_ROOT}/logs/run_$(date '+%Y%m%d_%H%M%S').log"

build_with_gold_input() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/input.json" ]]; then
    echo "[ZH-BOOT-OOF] Reusing ${out_dir}/input.json"
    return
  fi
  "${PYTHON_BIN}" "${WRAPPER_DIR}/build_gold_bootstrap_inputs.py" \
    --repo-dir "${REPRO_REPO}" \
    --mode dev-oof \
    --data-cwd "${BASE_PROJECT}" \
    --folded-csv "${BASE_PROJECT}/datasets/aligned_zh_folded.csv" \
    --output-path "${out_dir}/input.json" \
    --shot-num "${SHOT_NUM}" \
    --bootstrap-num "${BOOTSTRAP_NUM}" \
    --seed "${SEED}" \
    --metrics "${METRICS}" \
    --lang zh
}

build_without_gold_input() {
  local with_dir="$1"
  local without_dir="$2"
  mkdir -p "${without_dir}"
  if [[ -s "${without_dir}/input.json" ]]; then
    echo "[ZH-BOOT-OOF] Reusing ${without_dir}/input.json"
    return
  fi
  "${PYTHON_BIN}" "${WRAPPER_DIR}/make_no_gold_input.py" \
    --input-json "${with_dir}/input.json" \
    --output-json "${without_dir}/input.json" \
    --assert-no-gold
}

run_context() {
  local context="$1"
  local out_dir="$2"
  if [[ ! -s "${out_dir}/qwen30b.json" ]]; then
    "${PYTHON_BIN}" "${BASE_PROJECT}/model_runners/infer_qwen3.py" \
      --model_path "${QWEN_MODEL}" \
      --data_path "${out_dir}/input.json" \
      --file_name "${out_dir}/qwen30b.json" \
      --run_id "zh-bootstrap20x7-dev-oof-${context}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  else
    echo "[ZH-BOOT-OOF] Reusing ${out_dir}/qwen30b.json"
  fi

  "${PYTHON_BIN}" "${WRAPPER_DIR}/aggregate_gold_bootstrap_predictions.py" \
    --raw-json "${out_dir}/qwen30b.json" \
    --template-csv "${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_zh.csv" \
    --gold-csv "${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_zh.csv" \
    --eval-py "${BASE_PROJECT}/eval_zh.py" \
    --output-csv "${out_dir}/prediction.csv" \
    --all-runs-csv "${out_dir}/prediction_all_bootstraps.csv" \
    --score-json "${out_dir}/prediction.json" \
    --metrics "${METRICS}" \
    --lang zh
}

main() {
  cd "${BASE_PROJECT}"
  echo "[ZH-BOOT-OOF] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[ZH-BOOT-OOF] OUT_ROOT=${OUT_ROOT}"
  local with_dir="${OUT_ROOT}/with_gold_dev_oof_20x7"
  local no_dir="${OUT_ROOT}/without_gold_dev_oof_20x7"
  build_with_gold_input "${with_dir}"
  run_context with_gold "${with_dir}"
  build_without_gold_input "${with_dir}" "${no_dir}"
  run_context without_gold "${no_dir}"
  echo "[ZH-BOOT-OOF] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee -a "${LOG_FILE}"
