#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/data/public_models/Qwen3-8B-text}"
LANG_MODE="${LANG_MODE:-en}"
INFER_PATH="${INFER_PATH:-datasets/mediqa-eval-2026-valid-aligned.csv}"
VALID_PATH="${VALID_PATH:-datasets/mediqa-eval-2026-valid_1rater_en.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
EXP_SUFFIX="${EXP_SUFFIX:-_repro_$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-rag_gold_qwen3_8b${EXP_SUFFIX}_ALL}"
RESULT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
LOG_DIR="${RESULT_DIR}/logs"
LOG_FILE="${LOG_DIR}/reproduce_rag.log"
INPUT_JSON="${RESULT_DIR}/rag_input.json"
RAW_JSON="${RESULT_DIR}/qwen.json"
PRED_CSV="${RESULT_DIR}/prediction.csv"
SCORES_JSON="${RESULT_DIR}/prediction.json"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
SAMPLE_N="${SAMPLE_N:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

mkdir -p "${LOG_DIR}"

MAKE_INPUT_ARGS=(
  --infer_path "${INFER_PATH}"
  --output_path "${INPUT_JSON}"
  --metrics "${METRICS}"
  --lang "${LANG_MODE}"
)

if [[ -n "${SAMPLE_N}" ]]; then
  MAKE_INPUT_ARGS+=(--sample_n "${SAMPLE_N}")
fi

{
  echo "[RAG REPRO] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[RAG REPRO] Python: ${PYTHON_BIN}"
  echo "[RAG REPRO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[RAG REPRO] Model: ${MODEL_PATH}"
  echo "[RAG REPRO] Lang: ${LANG_MODE}"
  echo "[RAG REPRO] Result dir: ${RESULT_DIR}"

  "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["pandas", "torch", "transformers", "draccus", "scipy"]
for mod_name in mods:
    mod = importlib.import_module(mod_name)
    print(f"[RAG REPRO] {mod_name} {getattr(mod, '__version__', '')}")
PY

  "${PYTHON_BIN}" exp/few_shot/scripts/make_rag_input_main.py "${MAKE_INPUT_ARGS[@]}"

  "${PYTHON_BIN}" model_runners/infer_qwen3.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${INPUT_JSON}" \
    --file_name "${RAW_JSON}" \
    --run_id "${EXP_NAME}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  "${PYTHON_BIN}" scripts/submission/make_submission_file.py \
    --pred_path "${RAW_JSON}" \
    --save_path "${PRED_CSV}" \
    --metrics "${METRICS}" \
    --template_path "${VALID_PATH}" \
    --en_only True

  "${PYTHON_BIN}" eval_en.py \
    "${VALID_PATH}" \
    "${PRED_CSV}" \
    "${SCORES_JSON}"

  "${PYTHON_BIN}" - "${SCORES_JSON}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    scores = json.load(f)
print("[RAG REPRO] Key scores")
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

  echo "[RAG REPRO] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG_FILE}"

echo "Saved log to ${LOG_FILE}"
