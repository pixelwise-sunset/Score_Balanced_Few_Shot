#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/liyuan/conda_envs/yolo/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STRATEGY_TYPE="${STRATEGY_TYPE:-dataset_specific}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
CSV_PATH="${CSV_PATH:-datasets/aligned_en_folded.csv}"
VALID_PATH="${VALID_PATH:-datasets/mediqa-eval-2026-valid_1rater_en.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
EXP_SUFFIX="${EXP_SUFFIX:-_repro_$(date +%Y%m%d_%H%M%S)}"
KEEP_MODEL="${KEEP_MODEL:-0}"

EXP_NAME="pubmedbert_${STRATEGY_TYPE}${EXP_SUFFIX}_ALL"
RESULT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
LOG_DIR="${RESULT_DIR}/logs"
LOG_FILE="${LOG_DIR}/reproduce_bert.log"

mkdir -p "${LOG_DIR}"

KEEP_MODEL_ARGS=()
if [[ "${KEEP_MODEL}" == "1" || "${KEEP_MODEL}" == "true" || "${KEEP_MODEL}" == "True" ]]; then
  KEEP_MODEL_ARGS+=(--keep_model)
fi

{
  echo "[BERT REPRO] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[BERT REPRO] Python: ${PYTHON_BIN}"
  echo "[BERT REPRO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[BERT REPRO] Result dir: ${RESULT_DIR}"
  echo "[BERT REPRO] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
  echo "[BERT REPRO] WANDB_MODE=${WANDB_MODE}"

  "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["pandas", "torch", "transformers", "sklearn", "scipy", "wandb"]
for mod_name in mods:
    mod = importlib.import_module(mod_name)
    print(f"[BERT REPRO] {mod_name} {getattr(mod, '__version__', '')}")
try:
    import datasets
    print(f"[BERT REPRO] datasets {datasets.__version__}")
except Exception as exc:
    print(f"[BERT REPRO] datasets unavailable; train_bert.py will use fallback: {exc}")
PY

  "${PYTHON_BIN}" train_bert.py \
    --strategy_type "${STRATEGY_TYPE}" \
    --exp_suffix "${EXP_SUFFIX}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --csv_path "${CSV_PATH}" \
    --official_valid_path "${VALID_PATH}" \
    --output_path "${OUTPUT_ROOT}" \
    "${KEEP_MODEL_ARGS[@]}"

  "${PYTHON_BIN}" eval_en.py \
    "${VALID_PATH}" \
    "${RESULT_DIR}/prediction.csv" \
    "${RESULT_DIR}/prediction.json"

  "${PYTHON_BIN}" - "${RESULT_DIR}/prediction.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    scores = json.load(f)
print("[BERT REPRO] Key scores")
for key in [
    "ALL-en-ALL-mean",
    "ALL-en-disagree_flag-mean",
    "ALL-en-completeness-mean",
    "ALL-en-factual-accuracy-mean",
    "ALL-en-relevance-mean",
    "ALL-en-writing-style-mean",
    "ALL-en-overall-mean",
]:
    print(f"{key}: {scores.get(key)}")
PY

  echo "[BERT REPRO] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG_FILE}"

echo "Saved log to ${LOG_FILE}"
