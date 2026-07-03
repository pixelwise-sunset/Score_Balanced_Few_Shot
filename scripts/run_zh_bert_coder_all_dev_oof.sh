#!/usr/bin/env bash
set -euo pipefail

BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
OUT_ROOT="${OUT_ROOT:-${BASE_PROJECT}/results/zh_bert_coder_all_main_20260618}"
MODEL_NAME="${MODEL_NAME:-GanjinZero/coder_all}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-32}"
LR="${LR:-2e-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${BASE_PROJECT}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}/logs"
LOG_FILE="${OUT_ROOT}/logs/run_$(date '+%Y%m%d_%H%M%S').log"

run_context() {
  local strategy="$1"
  local out_dir="$2"
  if [[ -s "${out_dir}/prediction.csv" && -s "${out_dir}/prediction.json" ]]; then
    echo "[ZH-BERT-OOF] Reusing ${out_dir}/prediction.csv"
    return
  fi
  "${PYTHON_BIN}" "${BASE_PROJECT}/scripts/bert/train_predict_zh_dev_oof_bert.py" \
    --strategy_type "${strategy}" \
    --train_csv "${BASE_PROJECT}/datasets/aligned_zh_folded.csv" \
    --official_valid_csv "${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_zh.csv" \
    --output_dir "${out_dir}" \
    --model_name "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --predict_batch_size "${PREDICT_BATCH_SIZE}" \
    --lr "${LR}"
}

main() {
  cd "${BASE_PROJECT}"
  echo "[ZH-BERT-OOF] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[ZH-BERT-OOF] OUT_ROOT=${OUT_ROOT}"
  echo "[ZH-BERT-OOF] MODEL_NAME=${MODEL_NAME}"
  run_context matched_with_gold "${OUT_ROOT}/with_gold"
  run_context matched_without_gold "${OUT_ROOT}/without_gold"
  echo "[ZH-BERT-OOF] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee -a "${LOG_FILE}"
