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

RUN_ID="${RUN_ID:-qwen-gold-bootstrap-oof-8b}"
QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-8B-text}"
FOLD_INPUT="${FOLD_INPUT:-datasets/aligned_en_folded.csv}"
VALID_PATH="${VALID_PATH:-datasets/mediqa-eval-2026-valid_1rater_en.csv}"
TRUE_CSV="${TRUE_CSV:-datasets/mediqa-eval-2026-valid.csv}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SAMPLE_N_PER_FOLD="${SAMPLE_N_PER_FOLD:-}"

RUN_DIR="exp/few_shot/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/reproduce_qwen_gold_bootstrap_oof.log"
FINAL_CSV="${RUN_DIR}/prediction.csv"
FINAL_SCORE_JSON="${RUN_DIR}/prediction.json"
FINAL_AGG_SCORE_JSON="${RUN_DIR}/aggregate_score.json"

mkdir -p "${LOG_DIR}"

{
  echo "[FEWSHOT OOF] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[FEWSHOT OOF] Python: ${PYTHON_BIN}"
  echo "[FEWSHOT OOF] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[FEWSHOT OOF] Model: ${QWEN_MODEL}"
  echo "[FEWSHOT OOF] Run dir: ${RUN_DIR}"
  echo "[FEWSHOT OOF] Folds: ${FOLDS}"
  echo "[FEWSHOT OOF] Shot num: ${SHOT_NUM}"
  echo "[FEWSHOT OOF] Bootstrap num: ${BOOTSTRAP_NUM}"

  "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["pandas", "torch", "transformers", "draccus", "scipy", "datasets"]
for mod_name in mods:
    mod = importlib.import_module(mod_name)
    print(f"[FEWSHOT OOF] {mod_name} {getattr(mod, '__version__', '')}")
PY

  for fold in ${FOLDS}; do
    FOLD_DIR="${RUN_DIR}/fold_${fold}"
    SPLIT_DIR="${FOLD_DIR}/splits"
    INPUT_JSON="${FOLD_DIR}/input.json"
    RAW_JSON="${FOLD_DIR}/qwen.json"
    AVG_CSV="${FOLD_DIR}/prediction.csv"
    AGG_SCORE_JSON="${FOLD_DIR}/aggregate_score.json"

    mkdir -p "${SPLIT_DIR}"

    echo "[FEWSHOT OOF] Preparing fold ${fold}"
    "${PYTHON_BIN}" exp/few_shot/scripts/prepare_few_shot_splits.py \
      --input_path "${FOLD_INPUT}" \
      --output_dir "${SPLIT_DIR}" \
      --valid_fold "${fold}" \
      --en_only

    BOOTSTRAP_ARGS=(
      --bs_sample_from "${SPLIT_DIR}/train.csv"
      --infer_path "${SPLIT_DIR}/val.csv"
      --output_path "${INPUT_JSON}"
      --shot_num "${SHOT_NUM}"
      --bootstrap_num "${BOOTSTRAP_NUM}"
      --metrics "${METRICS}"
      --en_only True
      --sample_with_replacement True
    )
    if [[ -n "${SAMPLE_N_PER_FOLD}" ]]; then
      BOOTSTRAP_ARGS+=(--sample_n "${SAMPLE_N_PER_FOLD}")
    fi
    "${PYTHON_BIN}" exp/few_shot/scripts/strategy/bootstrap_gold_shots.py "${BOOTSTRAP_ARGS[@]}"

    echo "[FEWSHOT OOF] Inference fold ${fold}"
    "${PYTHON_BIN}" model_runners/infer_qwen3.py \
      --model_path "${QWEN_MODEL}" \
      --data_path "${INPUT_JSON}" \
      --file_name "${RAW_JSON}" \
      --run_id "${RUN_ID}_fold_${fold}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"

    echo "[FEWSHOT OOF] Aggregating fold ${fold}"
    "${PYTHON_BIN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
      --prediction_path "${RAW_JSON}" \
      --true_path "${TRUE_CSV}" \
      --output_csv "${AVG_CSV}" \
      --score_json "${AGG_SCORE_JSON}" \
      --metrics "${METRICS}" \
      --markdown
  done

  echo "[FEWSHOT OOF] Merging fold predictions"
  "${PYTHON_BIN}" - "${RUN_DIR}" "${FINAL_CSV}" <<'PY'
import glob
import os
import sys
import pandas as pd

run_dir = sys.argv[1]
output_csv = sys.argv[2]
paths = sorted(glob.glob(os.path.join(run_dir, "fold_*", "prediction.csv")))
if not paths:
    raise SystemExit("No fold prediction.csv files found")
dfs = [pd.read_csv(path) for path in paths]
merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates(
    subset=["dataset", "encounter_id", "lang", "candidate", "candidate_author_id", "metric"],
    keep="first",
)
merged.to_csv(output_csv, index=False)
print(f"Saved merged prediction CSV to {output_csv}")
print(f"Merged rows: {len(merged)}")
print(f"Metric rows: {merged['metric'].value_counts().to_dict()}")
PY

  "${PYTHON_BIN}" eval_en.py \
    "${VALID_PATH}" \
    "${FINAL_CSV}" \
    "${FINAL_SCORE_JSON}"

  "${PYTHON_BIN}" - "${RUN_DIR}" "${FINAL_AGG_SCORE_JSON}" "${METRICS}" <<'PY'
import ast
import glob
import json
import os
import sys

run_dir = sys.argv[1]
output_path = sys.argv[2]
metrics = ast.literal_eval(sys.argv[3])
paths = sorted(glob.glob(os.path.join(run_dir, "fold_*", "aggregate_score.json")))
if not paths:
    raise SystemExit("No fold aggregate_score.json files found")
per_metric = {metric: [] for metric in metrics}
overall = []
for path in paths:
    with open(path, encoding="utf-8") as f:
        scores = json.load(f)
    for metric in metrics:
        if metric in scores:
            per_metric[metric].append(scores[metric])
    if "ALL_en_ALL_mean" in scores:
        overall.append(scores["ALL_en_ALL_mean"])

summary = {
    metric: (sum(values) / len(values) if values else None)
    for metric, values in per_metric.items()
}
summary["ALL_en_ALL_mean"] = sum(overall) / len(overall) if overall else None
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Saved fold-mean aggregate score to {output_path}")
PY

  "${PYTHON_BIN}" - "${FINAL_AGG_SCORE_JSON}" "${FINAL_SCORE_JSON}" <<'PY'
import json
import sys
agg_path = sys.argv[1]
eval_path = sys.argv[2]
with open(agg_path, encoding="utf-8") as f:
    agg_scores = json.load(f)
with open(eval_path, encoding="utf-8") as f:
    eval_scores = json.load(f)
print("[FEWSHOT OOF] Fold-mean aggregate score")
for key in [
    "ALL_en_ALL_mean",
    "disagree_flag",
    "completeness",
    "factual-accuracy",
    "relevance",
    "writing-style",
    "overall",
]:
    print(f"{key}: {agg_scores.get(key)}")
print("[FEWSHOT OOF] Full-dev eval_en score")
for key in [
    "ALL-en-ALL-mean",
    "ALL-en-disagree_flag-mean",
    "ALL-en-completeness-mean",
    "ALL-en-factual-accuracy-mean",
    "ALL-en-relevance-mean",
    "ALL-en-writing-style-mean",
    "ALL-en-overall-mean",
]:
    print(f"{key}: {eval_scores.get(key)}")
PY

  echo "[FEWSHOT OOF] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG_FILE}"

echo "Saved log to ${LOG_FILE}"
