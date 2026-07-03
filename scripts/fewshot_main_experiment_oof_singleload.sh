#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

STRATEGY="${STRATEGY:-similarity_mmr}"
SHOT_NUM="${SHOT_NUM:-20}"
RUN_ID="${RUN_ID:-qwen-gold-${STRATEGY}-shot${SHOT_NUM}-oof-singleload}"
QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/public_models/bge-m3}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-sentence_transformer}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
FOLD_INPUT="${FOLD_INPUT:-datasets/aligned_en_folded.csv}"
VALID_PATH="${VALID_PATH:-datasets/mediqa-eval-2026-valid_1rater_en.csv}"
TRUE_CSV="${TRUE_CSV:-datasets/mediqa-eval-2026-valid.csv}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SAMPLE_N_PER_FOLD="${SAMPLE_N_PER_FOLD:-}"
KEEP_EXISTING_INPUT="${KEEP_EXISTING_INPUT:-1}"
KEEP_EXISTING_RAW="${KEEP_EXISTING_RAW:-1}"

RUN_DIR="exp/few_shot/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/main_experiment_oof_singleload.log"
MERGED_INPUT_JSON="${RUN_DIR}/input_all_folds.json"
RAW_JSON="${RUN_DIR}/qwen.json"
FINAL_CSV="${RUN_DIR}/prediction.csv"
FINAL_SCORE_JSON="${RUN_DIR}/prediction.json"
FINAL_AGG_SCORE_JSON="${RUN_DIR}/aggregate_score.json"

mkdir -p "${LOG_DIR}"

{
  echo "[MAIN OOF SINGLELOAD] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[MAIN OOF SINGLELOAD] Strategy: ${STRATEGY}"
  echo "[MAIN OOF SINGLELOAD] Shot num: ${SHOT_NUM}"
  echo "[MAIN OOF SINGLELOAD] Run dir: ${RUN_DIR}"
  echo "[MAIN OOF SINGLELOAD] Model: ${QWEN_MODEL}"
  echo "[MAIN OOF SINGLELOAD] Embedding model: ${EMBEDDING_MODEL}"
  echo "[MAIN OOF SINGLELOAD] Embedding backend: ${EMBEDDING_BACKEND}"
  echo "[MAIN OOF SINGLELOAD] Folds: ${FOLDS}"

  "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["pandas", "torch", "transformers", "sentence_transformers", "sklearn", "faiss"]
for mod_name in mods:
    try:
        mod = importlib.import_module(mod_name)
        print(f"[MAIN OOF SINGLELOAD] {mod_name} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"[MAIN OOF SINGLELOAD] {mod_name} import failed: {exc}")
PY

  for fold in ${FOLDS}; do
    FOLD_DIR="${RUN_DIR}/fold_${fold}"
    SPLIT_DIR="${FOLD_DIR}/splits"
    INPUT_JSON="${FOLD_DIR}/input.json"

    if [[ "${KEEP_EXISTING_INPUT}" == "1" && -f "${INPUT_JSON}" ]]; then
      echo "[MAIN OOF SINGLELOAD] Fold ${fold} input exists, skipping prompt build"
      continue
    fi

    mkdir -p "${SPLIT_DIR}"

    echo "[MAIN OOF SINGLELOAD] Preparing fold ${fold}"
    "${PYTHON_BIN}" exp/few_shot/scripts/prepare_few_shot_splits.py \
      --input_path "${FOLD_INPUT}" \
      --output_dir "${SPLIT_DIR}" \
      --valid_fold "${fold}" \
      --en_only

    INPUT_ARGS=()
    if [[ -n "${SAMPLE_N_PER_FOLD}" ]]; then
      INPUT_ARGS+=(--sample_n "${SAMPLE_N_PER_FOLD}")
    fi

    if [[ "${STRATEGY}" == "random_bootstrap" ]]; then
      EFFECTIVE_BOOTSTRAP="${BOOTSTRAP_NUM}"
      if [[ "${SHOT_NUM}" == "0" ]]; then
        EFFECTIVE_BOOTSTRAP="1"
      fi
      echo "[MAIN OOF SINGLELOAD] Building random bootstrap input fold ${fold}"
      "${PYTHON_BIN}" exp/few_shot/scripts/strategy/bootstrap_gold_shots.py \
        --bs_sample_from "${SPLIT_DIR}/train.csv" \
        --infer_path "${SPLIT_DIR}/val.csv" \
        --output_path "${INPUT_JSON}" \
        --shot_num "${SHOT_NUM}" \
        --bootstrap_num "${EFFECTIVE_BOOTSTRAP}" \
        --metrics "${METRICS}" \
        --en_only True \
        --sample_with_replacement True \
        "${INPUT_ARGS[@]}"
    elif [[ "${STRATEGY}" == "similarity_topk" || "${STRATEGY}" == "similarity_weighted" || "${STRATEGY}" == "similarity_mmr" ]]; then
      echo "[MAIN OOF SINGLELOAD] Building ${STRATEGY} input fold ${fold}"
      "${PYTHON_BIN}" exp/few_shot/scripts/strategy/similarity_gold_shots.py \
        --bs_sample_from "${SPLIT_DIR}/train.csv" \
        --infer_path "${SPLIT_DIR}/val.csv" \
        --output_path "${INPUT_JSON}" \
        --shot_num "${SHOT_NUM}" \
        --metrics "${METRICS}" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --embedding_backend "${EMBEDDING_BACKEND}" \
        --embedding_max_length "${EMBEDDING_MAX_LENGTH}" \
        --strategy "${STRATEGY}" \
        --bootstrap_num "${BOOTSTRAP_NUM}" \
        --en_only True \
        "${INPUT_ARGS[@]}"
    else
      echo "Unknown STRATEGY=${STRATEGY}" >&2
      exit 2
    fi
  done

  echo "[MAIN OOF SINGLELOAD] Merging fold inputs"
  "${PYTHON_BIN}" - "${RUN_DIR}" "${MERGED_INPUT_JSON}" <<'PY'
import glob
import json
import os
import sys

run_dir = sys.argv[1]
output_path = sys.argv[2]
paths = sorted(glob.glob(os.path.join(run_dir, "fold_*", "input.json")))
if not paths:
    raise SystemExit("No fold input.json files found")
merged = []
for path in paths:
    with open(path) as f:
        data = json.load(f)
    for item in data:
        item = dict(item)
        item["oof_fold"] = os.path.basename(os.path.dirname(path)).replace("fold_", "")
        merged.append(item)
with open(output_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"Saved {len(merged)} prompt items to {output_path}")
PY

  if [[ "${KEEP_EXISTING_RAW}" == "1" && -f "${RAW_JSON}" ]]; then
    echo "[MAIN OOF SINGLELOAD] Raw prediction exists, skipping inference"
  else
    echo "[MAIN OOF SINGLELOAD] Running single-load Qwen inference"
    "${PYTHON_BIN}" model_runners/infer_qwen3.py \
      --model_path "${QWEN_MODEL}" \
      --data_path "${MERGED_INPUT_JSON}" \
      --file_name "${RAW_JSON}" \
      --run_id "${RUN_ID}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  fi

  echo "[MAIN OOF SINGLELOAD] Aggregating predictions"
  "${PYTHON_BIN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
    --prediction_path "${RAW_JSON}" \
    --true_path "${TRUE_CSV}" \
    --output_csv "${FINAL_CSV}" \
    --score_json "${FINAL_AGG_SCORE_JSON}" \
    --metrics "${METRICS}" \
    --markdown

  "${PYTHON_BIN}" eval_en.py \
    "${VALID_PATH}" \
    "${FINAL_CSV}" \
    "${FINAL_SCORE_JSON}"

  echo "[MAIN OOF SINGLELOAD] Full-dev eval_en score"
  "${PYTHON_BIN}" - "${FINAL_SCORE_JSON}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path) as f:
    scores = json.load(f)
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

  echo "[MAIN OOF SINGLELOAD] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Saved log to ${LOG_FILE}"
} 2>&1 | tee "${LOG_FILE}"
