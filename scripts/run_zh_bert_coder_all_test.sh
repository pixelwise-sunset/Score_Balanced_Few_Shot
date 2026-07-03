#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BERT="${PYTHON_BERT:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/workspace/models}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/zh_bert_coder_all_test}"
ASSET_DIR="${ASSET_DIR:-datasets/test_assets}"
TRAIN_CSV="${TRAIN_CSV:-datasets/aligned_zh_folded.csv}"
TEST_GOLD="${TEST_GOLD:-${ASSET_DIR}/test_gold_zh.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-${ASSET_DIR}/test_template_zh.csv}"
TEST_ALIGNED="${TEST_ALIGNED:-${ASSET_DIR}/test_aligned_zh.csv}"
MODEL_NAME="${MODEL_NAME:-GanjinZero/coder_all}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-32}"
LR="${LR:-2e-5}"

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/run_zh_bert_coder_all_$(date +%Y%m%d_%H%M%S).log"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ZH BERT] Missing required file: ${path}" >&2
    exit 1
  fi
}

build_assets_if_needed() {
  if [[ -s "${TEST_ALIGNED}" && -s "${TEST_TEMPLATE}" && -s "${TEST_GOLD}" ]]; then
    echo "[ZH BERT] Chinese test assets exist; skipping build"
    return
  fi
  "${PYTHON_BERT}" scripts/testset/build_chinese_test_assets.py \
    --test_gold datasets/mediqa-eval-2026-test.csv \
    --output_dir "${ASSET_DIR}"
}

run_bert() {
  local strategy="$1"
  local out_dir="${OUTPUT_ROOT}/bert_${strategy}"
  if [[ -s "${out_dir}/prediction.json" ]]; then
    echo "[ZH BERT] ${strategy} exists; skipping"
    return
  fi
  "${PYTHON_BERT}" scripts/bert/train_predict_zh_test_bert.py \
    --strategy_type "${strategy}" \
    --train_csv "${TRAIN_CSV}" \
    --test_aligned_csv "${TEST_ALIGNED}" \
    --test_template_csv "${TEST_TEMPLATE}" \
    --test_gold_csv "${TEST_GOLD}" \
    --output_dir "${out_dir}" \
    --model_name "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --predict_batch_size "${PREDICT_BATCH_SIZE}" \
    --lr "${LR}"
}

make_table() {
  "${PYTHON_BERT}" - <<PY
import json
import pandas as pd

entries = [
    ("BERT w/ gold", "${OUTPUT_ROOT}/bert_matched_with_gold/prediction.json"),
    ("BERT w/o gold", "${OUTPUT_ROOT}/bert_matched_without_gold/prediction.json"),
]
rows = []
for method, path in entries:
    scores = json.load(open(path, encoding="utf-8"))
    rows.append({
        "method": method,
        "score_json": path,
        "all_zh_all_mean": scores.get("ALL-zh-ALL-mean"),
        "factual_consistency_wgold_mean": scores.get("ALL-zh-factual-consistency-wgold-mean"),
        "writing_style_mean": scores.get("ALL-zh-writing-style-mean"),
    })
out = "${OUTPUT_ROOT}/table_zh_bert_coder_all.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"[ZH BERT] Saved {out}")
print(pd.DataFrame(rows).to_string(index=False))
PY
}

main() {
  {
    echo "[ZH BERT] Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[ZH BERT] OUTPUT_ROOT=${OUTPUT_ROOT}"
    echo "[ZH BERT] MODEL_NAME=${MODEL_NAME}"
    echo "[ZH BERT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[ZH BERT] EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} LR=${LR}"
    build_assets_if_needed
    require_file "${TRAIN_CSV}"
    require_file "${TEST_ALIGNED}"
    require_file "${TEST_TEMPLATE}"
    require_file "${TEST_GOLD}"
    run_bert matched_with_gold
    run_bert matched_without_gold
    make_table
    echo "[ZH BERT] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee "${LOG_FILE}"
}

main "$@"
