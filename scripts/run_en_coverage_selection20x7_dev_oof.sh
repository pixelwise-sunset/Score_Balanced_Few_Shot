#!/usr/bin/env bash
set -euo pipefail

BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
REPRO_REPO="${REPRO_REPO:-${BASE_PROJECT}}"
PYTHON_QWEN="${PYTHON_QWEN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
STRATEGY="${STRATEGY:-metric_bin_coverage}"
METHOD_KEY="${METHOD_KEY:-metric_bin_coverage20x7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_PROJECT}/results/en_shot_selection_probabilistic20x7/dev_oof/${METHOD_KEY}}"
FOLDED_CSV="${FOLDED_CSV:-${BASE_PROJECT}/datasets/aligned_en_folded.csv}"
DEV_GOLD="${DEV_GOLD:-${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_en.csv}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-hf_mean_pool}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPRO_REPO}:${BASE_PROJECT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/with_gold" "${OUTPUT_ROOT}/no_gold"
LOG_FILE="${OUTPUT_ROOT}/logs/run_$(date '+%Y%m%d_%H%M%S').log"

build_inputs() {
  if [[ ! -s "${OUTPUT_ROOT}/with_gold/input.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/shot_selection/build_coverage_selection_inputs.py" \
      --repo-dir "${REPRO_REPO}" \
      --data-cwd "${BASE_PROJECT}" \
      --mode dev_oof \
      --strategy "${STRATEGY}" \
      --folded-csv "${FOLDED_CSV}" \
      --output-json "${OUTPUT_ROOT}/with_gold/input.json" \
      --metrics "${METRICS}" \
      --shot-num "${SHOT_NUM}" \
      --bootstrap-num "${BOOTSTRAP_NUM}" \
      --seed "${SEED}" \
      --lang en \
      --embedding-model "${EMBEDDING_MODEL}" \
      --embedding-backend "${EMBEDDING_BACKEND}" \
      --embedding-max-length "${EMBEDDING_MAX_LENGTH}"
  else
    echo "[COVERAGE-DEV-OOF] Reusing ${OUTPUT_ROOT}/with_gold/input.json"
  fi

  if [[ ! -s "${OUTPUT_ROOT}/no_gold/input.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/paper_repro/make_no_gold_input.py" \
      --input-json "${OUTPUT_ROOT}/with_gold/input.json" \
      --output-json "${OUTPUT_ROOT}/no_gold/input.json" \
      --assert-no-gold
  else
    echo "[COVERAGE-DEV-OOF] Reusing ${OUTPUT_ROOT}/no_gold/input.json"
  fi
}

run_context() {
  local context="$1"
  local ctx_dir="${OUTPUT_ROOT}/${context}"
  local raw_json="${ctx_dir}/qwen30b.json"

  if [[ ! -s "${raw_json}" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/model_runners/infer_qwen3.py" \
      --model_path "${QWEN_MODEL}" \
      --data_path "${ctx_dir}/input.json" \
      --file_name "${raw_json}" \
      --run_id "${METHOD_KEY}-dev-oof-${context}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  else
    echo "[COVERAGE-DEV-OOF] Reusing ${raw_json}"
  fi

  "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/paper_repro/aggregate_gold_bootstrap_predictions.py" \
    --raw-json "${raw_json}" \
    --template-csv "${DEV_GOLD}" \
    --gold-csv "${DEV_GOLD}" \
    --eval-py "${BASE_PROJECT}/eval_en.py" \
    --output-csv "${ctx_dir}/prediction.csv" \
    --all-runs-csv "${ctx_dir}/prediction_all_bootstraps.csv" \
    --score-json "${ctx_dir}/prediction.json" \
    --lang en
}

main() {
  cd "${BASE_PROJECT}"
  echo "[COVERAGE-DEV-OOF] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[COVERAGE-DEV-OOF] STRATEGY=${STRATEGY}"
  echo "[COVERAGE-DEV-OOF] METHOD_KEY=${METHOD_KEY}"
  echo "[COVERAGE-DEV-OOF] OUTPUT_ROOT=${OUTPUT_ROOT}"
  build_inputs
  run_context with_gold
  run_context no_gold
  echo "[COVERAGE-DEV-OOF] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee -a "${LOG_FILE}"
