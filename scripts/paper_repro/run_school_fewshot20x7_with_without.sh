#!/usr/bin/env bash
set -euo pipefail

REPRO_REPO="${REPRO_REPO:-/workspace/MEDIQA-SUATBMI-ENSEMBLE-repro}"
WRAPPER_DIR="${WRAPPER_DIR:-/workspace/MediQA/scripts/paper_repro}"
BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
OUT_ROOT="${OUT_ROOT:-/workspace/MediQA/results/fewshot_bootstrap20x7_original_20260617}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
RUN_DEV="${RUN_DEV:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_WITH_GOLD="${RUN_WITH_GOLD:-1}"
RUN_WITHOUT_GOLD="${RUN_WITHOUT_GOLD:-1}"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPRO_REPO}:${BASE_PROJECT}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}/logs"
LOG_FILE="${OUT_ROOT}/logs/run_$(date '+%Y%m%d_%H%M%S').log"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[FEWSHOT][ERROR] Missing required file: ${path}" >&2
    exit 2
  fi
}

build_with_gold_input() {
  local mode="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/input.json" ]]; then
    echo "[FEWSHOT] Reusing ${out_dir}/input.json"
    return
  fi

  if [[ "${mode}" == "dev" ]]; then
    "${PYTHON_BIN}" "${WRAPPER_DIR}/build_gold_bootstrap_inputs.py" \
      --repo-dir "${REPRO_REPO}" \
      --mode dev-oof \
      --data-cwd "${BASE_PROJECT}" \
      --folded-csv "${BASE_PROJECT}/datasets/aligned_en_folded.csv" \
      --output-path "${out_dir}/input.json" \
      --shot-num "${SHOT_NUM}" \
      --bootstrap-num "${BOOTSTRAP_NUM}" \
      --seed "${SEED}"
  else
    "${PYTHON_BIN}" "${WRAPPER_DIR}/build_gold_bootstrap_inputs.py" \
      --repo-dir "${REPRO_REPO}" \
      --mode test \
      --data-cwd "${BASE_PROJECT}" \
      --shot-csv "${BASE_PROJECT}/datasets/aligned_en_folded.csv" \
      --infer-csv "${BASE_PROJECT}/datasets/test_assets/test_aligned_en.csv" \
      --output-path "${out_dir}/input.json" \
      --shot-num "${SHOT_NUM}" \
      --bootstrap-num "${BOOTSTRAP_NUM}" \
      --seed "${SEED}"
  fi
}

build_without_gold_input() {
  local with_dir="$1"
  local without_dir="$2"
  mkdir -p "${without_dir}"
  if [[ -s "${without_dir}/input.json" ]]; then
    echo "[FEWSHOT] Reusing ${without_dir}/input.json"
    return
  fi
  "${PYTHON_BIN}" "${WRAPPER_DIR}/make_no_gold_input.py" \
    --input-json "${with_dir}/input.json" \
    --output-json "${without_dir}/input.json" \
    --assert-no-gold
}

infer_raw() {
  local run_id="$1"
  local out_dir="$2"
  if [[ -s "${out_dir}/qwen30b.json" ]]; then
    echo "[FEWSHOT] Reusing ${out_dir}/qwen30b.json"
    return
  fi
  "${PYTHON_BIN}" "${REPRO_REPO}/model_runners/infer_qwen3.py" \
    --model_path "${QWEN_MODEL}" \
    --data_path "${out_dir}/input.json" \
    --file_name "${out_dir}/qwen30b.json" \
    --run_id "${run_id}" \
    --device auto
}

aggregate_scores() {
  local mode="$1"
  local out_dir="$2"
  local template_csv
  local gold_csv
  if [[ "${mode}" == "dev" ]]; then
    template_csv="${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_en.csv"
    gold_csv="${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_en.csv"
  else
    template_csv="${BASE_PROJECT}/datasets/test_assets/test_gold_en.csv"
    gold_csv="${BASE_PROJECT}/datasets/test_assets/test_gold_en.csv"
  fi
  "${PYTHON_BIN}" "${WRAPPER_DIR}/aggregate_gold_bootstrap_predictions.py" \
    --raw-json "${out_dir}/qwen30b.json" \
    --template-csv "${template_csv}" \
    --gold-csv "${gold_csv}" \
    --eval-py "${BASE_PROJECT}/eval_en.py" \
    --output-csv "${out_dir}/prediction.csv" \
    --all-runs-csv "${out_dir}/prediction_all_bootstraps.csv" \
    --score-json "${out_dir}/prediction.json"
}

run_context() {
  local mode="$1"
  local context="$2"
  local out_dir="$3"
  echo "[FEWSHOT] Running ${mode} ${context}"
  infer_raw "fewshot-${context}-${mode}-20x7-original" "${out_dir}"
  aggregate_scores "${mode}" "${out_dir}"
}

main() {
  echo "[FEWSHOT] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[FEWSHOT] REPRO_REPO=${REPRO_REPO}"
  echo "[FEWSHOT] BASE_PROJECT=${BASE_PROJECT}"
  echo "[FEWSHOT] OUT_ROOT=${OUT_ROOT}"
  echo "[FEWSHOT] QWEN_MODEL=${QWEN_MODEL}"
  echo "[FEWSHOT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[FEWSHOT] SHOT_NUM=${SHOT_NUM} BOOTSTRAP_NUM=${BOOTSTRAP_NUM} SEED=${SEED}"

  require_file "${REPRO_REPO}/model_runners/infer_qwen3.py"
  require_file "${WRAPPER_DIR}/build_gold_bootstrap_inputs.py"
  require_file "${WRAPPER_DIR}/make_no_gold_input.py"
  require_file "${WRAPPER_DIR}/aggregate_gold_bootstrap_predictions.py"
  require_file "${BASE_PROJECT}/datasets/aligned_en_folded.csv"
  require_file "${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_en.csv"
  require_file "${BASE_PROJECT}/datasets/test_assets/test_aligned_en.csv"
  require_file "${BASE_PROJECT}/datasets/test_assets/test_gold_en.csv"

  cat > "${OUT_ROOT}/manifest.json" <<EOF
{
  "method": "original random bootstrap few-shot",
  "shot_num": ${SHOT_NUM},
  "bootstrap_num": ${BOOTSTRAP_NUM},
  "seed": ${SEED},
  "repro_repo": "${REPRO_REPO}",
  "base_project": "${BASE_PROJECT}",
  "qwen_model": "${QWEN_MODEL}",
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

  if [[ "${RUN_DEV}" == "1" ]]; then
    DEV_WITH="${OUT_ROOT}/with_gold_dev_oof_20x7"
    DEV_NO="${OUT_ROOT}/without_gold_dev_oof_20x7"
    build_with_gold_input dev "${DEV_WITH}"
    if [[ "${RUN_WITH_GOLD}" == "1" ]]; then
      run_context dev with_gold "${DEV_WITH}"
    fi
    if [[ "${RUN_WITHOUT_GOLD}" == "1" ]]; then
      build_without_gold_input "${DEV_WITH}" "${DEV_NO}"
      run_context dev without_gold "${DEV_NO}"
    fi
  fi

  if [[ "${RUN_TEST}" == "1" ]]; then
    TEST_WITH="${OUT_ROOT}/with_gold_test_20x7"
    TEST_NO="${OUT_ROOT}/without_gold_test_20x7"
    build_with_gold_input test "${TEST_WITH}"
    if [[ "${RUN_WITH_GOLD}" == "1" ]]; then
      run_context test with_gold "${TEST_WITH}"
    fi
    if [[ "${RUN_WITHOUT_GOLD}" == "1" ]]; then
      build_without_gold_input "${TEST_WITH}" "${TEST_NO}"
      run_context test without_gold "${TEST_NO}"
    fi
  fi

  echo "[FEWSHOT] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee -a "${LOG_FILE}"
