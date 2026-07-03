#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext}"
OUT_ROOT="${OUT_ROOT:-results/paper_style_bert_retrain_20260618}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-64}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"
KEEP_MODEL="${KEEP_MODEL:-1}"

mkdir -p "${OUT_ROOT}/logs"
RUN_LOG="${OUT_ROOT}/logs/run_$(date '+%Y%m%d_%H%M%S').log"

KEEP_MODEL_ARGS=()
if [[ "${KEEP_MODEL}" == "1" || "${KEEP_MODEL}" == "true" || "${KEEP_MODEL}" == "True" ]]; then
  KEEP_MODEL_ARGS+=(--keep_model)
fi

run_one() {
  local tag="$1"
  local use_gold="$2"
  local out_dir="${OUT_ROOT}/${tag}"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  {
    echo "[BERT] tag=${tag} use_gold=${use_gold} start $(date '+%F %T')"
    echo "[BERT] cuda=${CUDA_VISIBLE_DEVICES} python=${PYTHON_BIN}"
    echo "[BERT] model=${MODEL_NAME}"

    "${PYTHON_BIN}" scripts/bert/retrain_multitask_pubmedbert.py \
      --mode oof \
      --exp_name "paper_style_pubmedbert_${tag}" \
      --train_csv datasets/aligned_en_folded.csv \
      --official_valid_path datasets/mediqa-eval-2026-valid_1rater_en.csv \
      --output_dir "${out_dir}" \
      --model_name "${MODEL_NAME}" \
      --use_gold "${use_gold}" \
      --use_caption 1 \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --fp16 \
      "${KEEP_MODEL_ARGS[@]}"

    "${PYTHON_BIN}" eval_en.py \
      datasets/mediqa-eval-2026-valid_1rater_en.csv \
      "${out_dir}/prediction.csv" \
      "${out_dir}/prediction.json"

    "${PYTHON_BIN}" scripts/bert/retrain_multitask_pubmedbert.py \
      --mode test \
      --exp_name "paper_style_pubmedbert_${tag}" \
      --output_dir "${out_dir}" \
      --checkpoint_dir "${out_dir}" \
      --model_name "${MODEL_NAME}" \
      --use_gold "${use_gold}" \
      --use_caption 1 \
      --num_folds 5 \
      --test_aligned_path datasets/test_assets/test_aligned_en.csv \
      --test_template_path datasets/test_assets/test_template_en.csv \
      --test_output_path "${out_dir}/prediction_test.csv" \
      --predict_batch_size "${PREDICT_BATCH_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --fp16

    "${PYTHON_BIN}" eval_en.py \
      datasets/test_assets/test_gold_en.csv \
      "${out_dir}/prediction_test.csv" \
      "${out_dir}/prediction_test.json"

    "${PYTHON_BIN}" - "${out_dir}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
keys = [
    "ALL-en-ALL-mean",
    "ALL-en-disagree_flag-mean",
    "ALL-en-completeness-mean",
    "ALL-en-factual-accuracy-mean",
    "ALL-en-relevance-mean",
    "ALL-en-writing-style-mean",
    "ALL-en-overall-mean",
]
for name in ["prediction.json", "prediction_test.json"]:
    scores = json.load(open(out / name, encoding="utf-8"))
    print(f"[SCORES] {out / name}")
    for key in keys:
        print(f"  {key}: {scores.get(key)}")
PY

    echo "[BERT] tag=${tag} done $(date '+%F %T')"
  } 2>&1 | tee -a "${RUN_LOG}"
}

run_one "with_gold" 1
run_one "without_gold" 0

echo "[BERT] finished $(date '+%F %T')"
echo "[BERT] log=${RUN_LOG}"
