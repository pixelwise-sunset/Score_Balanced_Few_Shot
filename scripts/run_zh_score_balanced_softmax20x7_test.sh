#!/usr/bin/env bash
set -euo pipefail

BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
PYTHON_QWEN="${PYTHON_QWEN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_PROJECT}/results/zh_shot_selection_probabilistic20x7}"
RUN_DIR="${OUTPUT_ROOT}/runs/score_balanced_softmax20x7"
SHOT_POOL="${SHOT_POOL:-${BASE_PROJECT}/datasets/aligned_zh_folded.csv}"
TEST_ALIGNED="${TEST_ALIGNED:-${BASE_PROJECT}/datasets/test_assets/test_aligned_zh.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-${BASE_PROJECT}/datasets/test_assets/test_template_zh.csv}"
TEST_GOLD="${TEST_GOLD:-${BASE_PROJECT}/datasets/test_assets/test_gold_zh.csv}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-1.0}"
METRICS="${METRICS:-['factual-consistency-wgold','writing-style']}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${BASE_PROJECT}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}/with_gold" "${RUN_DIR}/no_gold" "${OUTPUT_ROOT}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/zh_score_balanced_softmax20x7_test_$(date '+%Y%m%d_%H%M%S').log"

build_inputs() {
  if [[ ! -s "${RUN_DIR}/with_gold/input.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/shot_selection/balanced_gold_shots.py" \
      --bs_sample_from "${SHOT_POOL}" \
      --infer_path "${TEST_ALIGNED}" \
      --output_path "${RUN_DIR}/with_gold/input.json" \
      --shot_num "${SHOT_NUM}" \
      --bootstrap_num "${BOOTSTRAP_NUM}" \
      --metrics "${METRICS}" \
      --strategy score_balanced_softmax \
      --softmax_temperature "${SOFTMAX_TEMPERATURE}" \
      --en_only False \
      --seed "${SEED}"
  else
    echo "[ZH-SBS-TEST] Reusing ${RUN_DIR}/with_gold/input.json"
  fi

  if [[ ! -s "${RUN_DIR}/no_gold/input.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/paper_repro/make_no_gold_input.py" \
      --input-json "${RUN_DIR}/with_gold/input.json" \
      --output-json "${RUN_DIR}/no_gold/input.json" \
      --assert-no-gold
  else
    echo "[ZH-SBS-TEST] Reusing ${RUN_DIR}/no_gold/input.json"
  fi
}

run_context() {
  local context="$1"
  local ctx_dir="${RUN_DIR}/${context}"
  local raw_json="${ctx_dir}/qwen30b.json"

  if [[ ! -s "${raw_json}" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/model_runners/infer_qwen3.py" \
      --model_path "${QWEN_MODEL}" \
      --data_path "${ctx_dir}/input.json" \
      --file_name "${raw_json}" \
      --run_id "zh-score-balanced-softmax20x7-test-${context}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  else
    echo "[ZH-SBS-TEST] Reusing ${raw_json}"
  fi

  "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/paper_repro/aggregate_gold_bootstrap_predictions.py" \
    --raw-json "${raw_json}" \
    --template-csv "${TEST_TEMPLATE}" \
    --gold-csv "${TEST_GOLD}" \
    --eval-py "${BASE_PROJECT}/eval_zh.py" \
    --output-csv "${ctx_dir}/prediction.csv" \
    --all-runs-csv "${ctx_dir}/prediction_all_bootstraps.csv" \
    --score-json "${ctx_dir}/prediction.json" \
    --metrics "${METRICS}" \
    --lang zh
}

main() {
  cd "${BASE_PROJECT}"
  echo "[ZH-SBS-TEST] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[ZH-SBS-TEST] RUN_DIR=${RUN_DIR}"
  build_inputs
  run_context with_gold
  run_context no_gold
  echo "[ZH-SBS-TEST] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee -a "${LOG_FILE}"
