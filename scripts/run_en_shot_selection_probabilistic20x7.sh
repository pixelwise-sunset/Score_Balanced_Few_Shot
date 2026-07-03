#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_QWEN="${PYTHON_QWEN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/en_shot_selection_probabilistic20x7}"
RUN_ROOT="${OUTPUT_ROOT}/runs"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

SHOT_POOL="${SHOT_POOL:-datasets/aligned_en_folded.csv}"
TEST_ALIGNED="${TEST_ALIGNED:-datasets/test_assets/test_aligned_en.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-datasets/test_assets/test_template_en.csv}"
TEST_GOLD="${TEST_GOLD:-datasets/test_assets/test_gold_en.csv}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-hf_mean_pool}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-1.0}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"
STRATEGIES="${STRATEGIES:-score_balanced_softmax20x7 mmr_sample20x7}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[PROB-SHOT] Missing file: ${path}" >&2
    exit 1
  fi
}

generate_with_gold() {
  local strategy="$1"
  local run_dir="$2"
  local input_json="${run_dir}/with_gold/input.json"
  mkdir -p "${run_dir}/with_gold" "${run_dir}/no_gold"
  if [[ -s "${input_json}" ]]; then
    echo "[PROB-SHOT] with-gold input exists; skipping ${input_json}"
    return
  fi

  case "${strategy}" in
    score_balanced_softmax20x7)
      "${PYTHON_QWEN}" scripts/shot_selection/balanced_gold_shots.py \
        --bs_sample_from "${SHOT_POOL}" \
        --infer_path "${TEST_ALIGNED}" \
        --output_path "${input_json}" \
        --shot_num "${SHOT_NUM}" \
        --bootstrap_num "${BOOTSTRAP_NUM}" \
        --metrics "${METRICS}" \
        --strategy score_balanced_softmax \
        --softmax_temperature "${SOFTMAX_TEMPERATURE}" \
        --en_only True \
        --seed "${SEED}"
      ;;
    metric_bin_coverage20x7)
      "${PYTHON_QWEN}" scripts/shot_selection/build_coverage_selection_inputs.py \
        --repo-dir "${ROOT_DIR}" \
        --data-cwd "${ROOT_DIR}" \
        --mode test \
        --strategy metric_bin_coverage \
        --shot-csv "${SHOT_POOL}" \
        --infer-csv "${TEST_ALIGNED}" \
        --output-json "${input_json}" \
        --shot-num "${SHOT_NUM}" \
        --bootstrap-num "${BOOTSTRAP_NUM}" \
        --metrics "${METRICS}" \
        --lang en \
        --seed "${SEED}"
      ;;
    per_sample_coverage_similarity20x7)
      "${PYTHON_QWEN}" scripts/shot_selection/build_coverage_selection_inputs.py \
        --repo-dir "${ROOT_DIR}" \
        --data-cwd "${ROOT_DIR}" \
        --mode test \
        --strategy per_sample_coverage_similarity \
        --shot-csv "${SHOT_POOL}" \
        --infer-csv "${TEST_ALIGNED}" \
        --output-json "${input_json}" \
        --shot-num "${SHOT_NUM}" \
        --bootstrap-num "${BOOTSTRAP_NUM}" \
        --metrics "${METRICS}" \
        --lang en \
        --seed "${SEED}" \
        --embedding-model "${EMBEDDING_MODEL}" \
        --embedding-backend "${EMBEDDING_BACKEND}" \
        --embedding-max-length "${EMBEDDING_MAX_LENGTH}"
      ;;
    mmr_sample20x7)
      "${PYTHON_QWEN}" exp/few_shot/scripts/strategy/similarity_gold_shots.py \
        --bs_sample_from "${SHOT_POOL}" \
        --infer_path "${TEST_ALIGNED}" \
        --output_path "${input_json}" \
        --shot_num "${SHOT_NUM}" \
        --bootstrap_num "${BOOTSTRAP_NUM}" \
        --metrics "${METRICS}" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --embedding_backend "${EMBEDDING_BACKEND}" \
        --embedding_max_length "${EMBEDDING_MAX_LENGTH}" \
        --strategy similarity_mmr \
        --en_only True \
        --seed "${SEED}"
      ;;
    *)
      echo "[PROB-SHOT] Unsupported strategy: ${strategy}" >&2
      exit 1
      ;;
  esac
}

generate_no_gold() {
  local run_dir="$1"
  local input_json="${run_dir}/no_gold/input.json"
  if [[ -s "${input_json}" ]]; then
    echo "[PROB-SHOT] no-gold input exists; skipping ${input_json}"
    return
  fi
  "${PYTHON_QWEN}" exp/few_shot/scripts/make_no_gold_input.py \
    --input_json "${run_dir}/with_gold/input.json" \
    --output_json "${input_json}" \
    --assert_no_gold
}

run_infer_eval() {
  local context="$1"
  local run_dir="$2"
  local run_id="$3"
  local method="$4"
  local ctx_dir="${run_dir}/${context}"
  local raw_json="${ctx_dir}/qwen30b.json"
  local pred_csv="${ctx_dir}/prediction.csv"
  local agg_json="${ctx_dir}/aggregate_score.json"
  local score_json="${ctx_dir}/prediction.json"

  if [[ ! -s "${raw_json}" ]]; then
    "${PYTHON_QWEN}" model_runners/infer_qwen3.py \
      --model_path "${QWEN_MODEL}" \
      --data_path "${ctx_dir}/input.json" \
      --file_name "${raw_json}" \
      --run_id "${run_id}" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  else
    echo "[PROB-SHOT] raw output exists; skipping ${raw_json}"
  fi

  if [[ ! -s "${pred_csv}" || ! -s "${agg_json}" ]]; then
    "${PYTHON_QWEN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
      --prediction_path "${raw_json}" \
      --true_path "${TEST_TEMPLATE}" \
      --output_csv "${pred_csv}" \
      --score_json "${agg_json}" \
      --metrics "${METRICS}" \
      --markdown
  else
    echo "[PROB-SHOT] aggregate exists; skipping ${pred_csv}"
  fi

  if [[ ! -s "${score_json}" ]]; then
    "${PYTHON_QWEN}" eval_en.py "${TEST_GOLD}" "${pred_csv}" "${score_json}"
  else
    echo "[PROB-SHOT] eval exists; skipping ${score_json}"
  fi

  if [[ ! -s "${ctx_dir}/per_bootstrap/per_bootstrap_summary.csv" ]]; then
    "${PYTHON_QWEN}" scripts/shot_selection/eval_per_bootstrap.py \
      --prediction_path "${raw_json}" \
      --true_path "${TEST_TEMPLATE}" \
      --output_dir "${ctx_dir}/per_bootstrap" \
      --metrics "${METRICS}" \
      --method "${method}" \
      --context "${context}" \
      --markdown
  else
    echo "[PROB-SHOT] per-bootstrap eval exists; skipping ${ctx_dir}/per_bootstrap"
  fi
}

main() {
  echo "[PROB-SHOT] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[PROB-SHOT] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[PROB-SHOT] STRATEGIES=${STRATEGIES}"
  echo "[PROB-SHOT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[PROB-SHOT] QWEN_MODEL=${QWEN_MODEL}"
  echo "[PROB-SHOT] SHOT_NUM=${SHOT_NUM} BOOTSTRAP_NUM=${BOOTSTRAP_NUM}"
  for path in "${SHOT_POOL}" "${TEST_ALIGNED}" "${TEST_TEMPLATE}" "${TEST_GOLD}"; do
    require_file "${path}"
  done

  for strategy in ${STRATEGIES}; do
    local run_dir="${RUN_ROOT}/${strategy}"
    echo "[PROB-SHOT] === ${strategy} ==="
    generate_with_gold "${strategy}" "${run_dir}"
    generate_no_gold "${run_dir}"
    run_infer_eval "with_gold" "${run_dir}" "en-prob-shot-${strategy}-with-gold" "${strategy}"
    run_infer_eval "no_gold" "${run_dir}" "en-prob-shot-${strategy}-no-gold" "${strategy}"
    "${PYTHON_QWEN}" scripts/shot_selection/summarize_probabilistic20x7.py \
      --output_root "${OUTPUT_ROOT}"
  done

  "${PYTHON_QWEN}" scripts/shot_selection/summarize_probabilistic20x7.py \
    --output_root "${OUTPUT_ROOT}"
  echo "[PROB-SHOT] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee "${LOG_FILE}"
