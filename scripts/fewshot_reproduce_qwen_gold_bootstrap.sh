#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-qwen-gold-bootstrap-repro}"
QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
FOLD_INPUT="${FOLD_INPUT:-datasets/aligned_en_folded.csv}"
DATA_DIR="${DATA_DIR:-exp/few_shot/datasets}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_DIR}/train.csv}"
INFER_CSV="${INFER_CSV:-${DATA_DIR}/val.csv}"
TRUE_CSV="${TRUE_CSV:-datasets/mediqa-eval-2026-valid.csv}"
VALID_PATH="${VALID_PATH:-datasets/mediqa-eval-2026-valid_1rater_en.csv}"
VALID_FOLD="${VALID_FOLD:-4}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"

RUN_DIR="exp/few_shot/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/reproduce_qwen_gold_bootstrap.log"
INPUT_JSON="${RUN_DIR}/input.json"
RAW_JSON="${RUN_DIR}/qwen30b.json"
AVG_CSV="${RUN_DIR}/qwen30b_avg.csv"
SCORE_JSON="${RUN_DIR}/qwen30b_avg_score.json"
STD_SCORE_JSON="${RUN_DIR}/qwen30b_avg_eval_en.json"

mkdir -p "${LOG_DIR}"

{
  echo "[FEWSHOT REPRO] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[FEWSHOT REPRO] Python: ${PYTHON_BIN}"
  echo "[FEWSHOT REPRO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[FEWSHOT REPRO] Model: ${QWEN_MODEL}"
  echo "[FEWSHOT REPRO] Run dir: ${RUN_DIR}"
  echo "[FEWSHOT REPRO] Valid fold: ${VALID_FOLD}"
  echo "[FEWSHOT REPRO] Shot num: ${SHOT_NUM}"
  echo "[FEWSHOT REPRO] Bootstrap num: ${BOOTSTRAP_NUM}"

  "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["pandas", "torch", "transformers", "draccus", "scipy", "datasets"]
for mod_name in mods:
    mod = importlib.import_module(mod_name)
    print(f"[FEWSHOT REPRO] {mod_name} {getattr(mod, '__version__', '')}")
PY

  "${PYTHON_BIN}" exp/few_shot/scripts/prepare_few_shot_splits.py \
    --input_path "${FOLD_INPUT}" \
    --output_dir "${DATA_DIR}" \
    --valid_fold "${VALID_FOLD}" \
    --en_only

  "${PYTHON_BIN}" exp/few_shot/scripts/strategy/bootstrap_gold_shots.py \
    --bs_sample_from "${TRAIN_CSV}" \
    --infer_path "${INFER_CSV}" \
    --output_path "${INPUT_JSON}" \
    --shot_num "${SHOT_NUM}" \
    --bootstrap_num "${BOOTSTRAP_NUM}" \
    --metrics "${METRICS}" \
    --en_only True \
    --sample_with_replacement True

  "${PYTHON_BIN}" model_runners/infer_qwen3.py \
    --model_path "${QWEN_MODEL}" \
    --data_path "${INPUT_JSON}" \
    --file_name "${RAW_JSON}" \
    --run_id "${RUN_ID}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  "${PYTHON_BIN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
    --prediction_path "${RAW_JSON}" \
    --true_path "${TRUE_CSV}" \
    --output_csv "${AVG_CSV}" \
    --score_json "${SCORE_JSON}" \
    --metrics "${METRICS}" \
    --markdown

  "${PYTHON_BIN}" eval_en.py \
    "${VALID_PATH}" \
    "${AVG_CSV}" \
    "${STD_SCORE_JSON}"

  "${PYTHON_BIN}" - "${SCORE_JSON}" "${STD_SCORE_JSON}" <<'PY'
import json
import sys
fewshot_score_path = sys.argv[1]
eval_en_score_path = sys.argv[2]
with open(fewshot_score_path, encoding="utf-8") as f:
    fewshot_scores = json.load(f)
with open(eval_en_score_path, encoding="utf-8") as f:
    eval_en_scores = json.load(f)
print("[FEWSHOT REPRO] Few-shot aggregate score")
for key in [
    "ALL_en_ALL_mean",
    "disagree_flag",
    "completeness",
    "factual-accuracy",
    "relevance",
    "writing-style",
    "overall",
]:
    print(f"{key}: {fewshot_scores.get(key)}")
print("[FEWSHOT REPRO] eval_en score")
for key in [
    "ALL-en-ALL-mean",
    "ALL-en-disagree_flag-mean",
    "ALL-en-completeness-mean",
    "ALL-en-factual-accuracy-mean",
    "ALL-en-relevance-mean",
    "ALL-en-writing-style-mean",
    "ALL-en-overall-mean",
]:
    print(f"{key}: {eval_en_scores.get(key)}")
PY

  echo "[FEWSHOT REPRO] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG_FILE}"

echo "Saved log to ${LOG_FILE}"
