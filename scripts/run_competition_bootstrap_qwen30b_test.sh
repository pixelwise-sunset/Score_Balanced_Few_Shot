#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_QWEN="${PYTHON_QWEN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/competition_bootstrap_qwen30b_test}"
SHOT_POOL="${SHOT_POOL:-datasets/aligned_en_folded.csv}"
TEST_ALIGNED="${TEST_ALIGNED:-datasets/test_assets/test_aligned_en.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-datasets/test_assets/test_template_en.csv}"
TEST_GOLD="${TEST_GOLD:-datasets/test_assets/test_gold_en.csv}"
QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"

WITH_GOLD_RUN_ID="${WITH_GOLD_RUN_ID:-competition-qwen30b-bootstrap20x7-test-with-gold}"
NO_GOLD_RUN_ID="${NO_GOLD_RUN_ID:-competition-qwen30b-bootstrap20x7-test-no-gold}"
OFFICIAL_SCORE_JSON="${OFFICIAL_SCORE_JSON:-results/competition_reproduction/prediction.json}"

WITH_DIR="exp/few_shot/runs/${WITH_GOLD_RUN_ID}"
NO_DIR="exp/few_shot/runs/${NO_GOLD_RUN_ID}"
mkdir -p "${OUTPUT_ROOT}/logs" "${WITH_DIR}/logs" "${NO_DIR}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/run_$(date +%Y%m%d_%H%M%S).log"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[BOOTSTRAP REPRO] Missing required file: ${path}" >&2
    exit 1
  fi
}

check_assets() {
  require_file "${SHOT_POOL}"
  require_file "${TEST_ALIGNED}"
  require_file "${TEST_TEMPLATE}"
  require_file "${TEST_GOLD}"
  require_file "${OFFICIAL_SCORE_JSON}"
  "${PYTHON_QWEN}" - <<PY
import pandas as pd
paths = {
    "shot_pool": "${SHOT_POOL}",
    "test_aligned": "${TEST_ALIGNED}",
    "test_template": "${TEST_TEMPLATE}",
    "test_gold": "${TEST_GOLD}",
}
for name, path in paths.items():
    df = pd.read_csv(path)
    print(f"[BOOTSTRAP REPRO] {name}: {path} rows={len(df)}")
assert len(pd.read_csv("${SHOT_POOL}")) == 2898
assert len(pd.read_csv("${TEST_ALIGNED}")) == 3474
assert len(pd.read_csv("${TEST_TEMPLATE}")) == 3474
assert len(pd.read_csv("${TEST_GOLD}")) == 6948
PY
}

generate_with_gold_input() {
  local input_json="${WITH_DIR}/input.json"
  if [[ -s "${input_json}" ]]; then
    echo "[BOOTSTRAP REPRO] with-gold input exists; skipping: ${input_json}"
    return
  fi
  "${PYTHON_QWEN}" exp/few_shot/scripts/strategy/bootstrap_gold_shots.py \
    --bs_sample_from "${SHOT_POOL}" \
    --infer_path "${TEST_ALIGNED}" \
    --output_path "${input_json}" \
    --shot_num "${SHOT_NUM}" \
    --bootstrap_num "${BOOTSTRAP_NUM}" \
    --metrics "${METRICS}" \
    --en_only True \
    --sample_with_replacement True \
    --seed "${SEED}"
}

run_qwen() {
  local run_id="$1"
  local run_dir="$2"
  local input_json="${run_dir}/input.json"
  local raw_json="${run_dir}/qwen30b.json"
  if [[ -s "${raw_json}" ]]; then
    echo "[BOOTSTRAP REPRO] raw Qwen output exists; skipping: ${raw_json}"
    return
  fi
  "${PYTHON_QWEN}" model_runners/infer_qwen3.py \
    --model_path "${QWEN_MODEL}" \
    --data_path "${input_json}" \
    --file_name "${raw_json}" \
    --run_id "${run_id}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

aggregate_and_eval() {
  local run_dir="$1"
  local raw_json="${run_dir}/qwen30b.json"
  local pred_csv="${run_dir}/prediction.csv"
  local agg_json="${run_dir}/aggregate_score.json"
  local score_json="${run_dir}/prediction.json"
  if [[ ! -s "${pred_csv}" || ! -s "${agg_json}" ]]; then
    "${PYTHON_QWEN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
      --prediction_path "${raw_json}" \
      --true_path "${TEST_TEMPLATE}" \
      --output_csv "${pred_csv}" \
      --score_json "${agg_json}" \
      --metrics "${METRICS}" \
      --markdown
  else
    echo "[BOOTSTRAP REPRO] aggregate output exists; skipping: ${pred_csv}"
  fi
  if [[ ! -s "${score_json}" ]]; then
    "${PYTHON_QWEN}" eval_en.py "${TEST_GOLD}" "${pred_csv}" "${score_json}"
  else
    echo "[BOOTSTRAP REPRO] eval output exists; skipping: ${score_json}"
  fi
}

generate_no_gold_input() {
  local input_json="${NO_DIR}/input.json"
  if [[ -s "${input_json}" ]]; then
    echo "[BOOTSTRAP REPRO] no-gold input exists; skipping: ${input_json}"
    return
  fi
  "${PYTHON_QWEN}" exp/few_shot/scripts/make_no_gold_input.py \
    --input_json "${WITH_DIR}/input.json" \
    --output_json "${input_json}" \
    --assert_no_gold
}

validate_outputs() {
  "${PYTHON_QWEN}" - <<PY
import json
import pandas as pd
from pathlib import Path

expected_prompts = 579 * ${BOOTSTRAP_NUM}
for name, run_dir, should_have_gold in [
    ("with_gold", "${WITH_DIR}", True),
    ("no_gold", "${NO_DIR}", False),
]:
    input_path = Path(run_dir) / "input.json"
    pred_path = Path(run_dir) / "prediction.csv"
    score_path = Path(run_dir) / "prediction.json"
    data = json.loads(input_path.read_text(encoding="utf-8"))
    assert len(data) == expected_prompts, (name, len(data), expected_prompts)
    joined = json.dumps(data, ensure_ascii=False).lower()
    assert ("gold responses" in joined) is should_have_gold, name
    pred = pd.read_csv(pred_path)
    assert len(pred) == 3474, (name, len(pred))
    assert pred["metric"].nunique() == 6, name
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    assert "ALL-en-ALL-mean" in scores, name
    print(f"[BOOTSTRAP REPRO] {name} ALL-en-ALL-mean={scores['ALL-en-ALL-mean']}")
PY
}

make_tables() {
  local with_entries=(
    --entry "Official Submission=${OFFICIAL_SCORE_JSON}"
    --entry "Bootstrap Few-shot=${WITH_DIR}/prediction.json"
  )
  local no_entries=(
    --entry "Official Submission=${OFFICIAL_SCORE_JSON}"
    --entry "Bootstrap Few-shot=${NO_DIR}/prediction.json"
  )

  [[ -f "results/main_tables_30b_shot20_pubmedbert_test/bert_matched_with_gold/prediction.json" ]] && \
    with_entries+=(--entry "BERT matched=results/main_tables_30b_shot20_pubmedbert_test/bert_matched_with_gold/prediction.json")
  [[ -f "results/main_tables_30b_shot20_pubmedbert_test/bert_historical_with_gold/prediction.json" ]] && \
    with_entries+=(--entry "BERT historical=results/main_tables_30b_shot20_pubmedbert_test/bert_historical_with_gold/prediction.json")

  [[ -f "results/main_tables_30b_shot20_pubmedbert_test/bert_matched_without_gold/prediction.json" ]] && \
    no_entries+=(--entry "BERT matched=results/main_tables_30b_shot20_pubmedbert_test/bert_matched_without_gold/prediction.json")
  [[ -f "results/main_tables_30b_shot20_pubmedbert_test/bert_historical_without_gold/prediction.json" ]] && \
    no_entries+=(--entry "BERT historical=results/main_tables_30b_shot20_pubmedbert_test/bert_historical_without_gold/prediction.json")
  [[ -f "exp/few_shot/runs/main30b-test-no-gold-rag-v2-pubmedbert-query-mmr-singleload/prediction.json" ]] && \
    no_entries+=(--entry "RAG v2 MMR=exp/few_shot/runs/main30b-test-no-gold-rag-v2-pubmedbert-query-mmr-singleload/prediction.json")

  "${PYTHON_QWEN}" scripts/summarize_main_tables.py \
    "${with_entries[@]}" \
    --output_csv "${OUTPUT_ROOT}/table_with_gold_bootstrap_reproduction.csv"
  "${PYTHON_QWEN}" scripts/summarize_main_tables.py \
    "${no_entries[@]}" \
    --output_csv "${OUTPUT_ROOT}/table_without_gold_bootstrap_reproduction.csv"
}

print_scores() {
  "${PYTHON_QWEN}" - <<PY
import json
for label, path in [
    ("Official Submission", "${OFFICIAL_SCORE_JSON}"),
    ("Bootstrap Few-shot with gold", "${WITH_DIR}/prediction.json"),
    ("Bootstrap Few-shot without gold", "${NO_DIR}/prediction.json"),
]:
    scores = json.load(open(path, encoding="utf-8"))
    print(f"[BOOTSTRAP REPRO] {label}")
    for key in [
        "ALL-en-ALL-mean",
        "ALL-en-disagree_flag-mean",
        "ALL-en-completeness-mean",
        "ALL-en-factual-accuracy-mean",
        "ALL-en-relevance-mean",
        "ALL-en-writing-style-mean",
        "ALL-en-overall-mean",
    ]:
        print(f"  {key}: {scores.get(key)}")
PY
}

main() {
  echo "[BOOTSTRAP REPRO] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[BOOTSTRAP REPRO] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[BOOTSTRAP REPRO] QWEN_MODEL=${QWEN_MODEL}"
  echo "[BOOTSTRAP REPRO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[BOOTSTRAP REPRO] SHOT_NUM=${SHOT_NUM} BOOTSTRAP_NUM=${BOOTSTRAP_NUM} SEED=${SEED}"
  check_assets
  generate_with_gold_input
  run_qwen "${WITH_GOLD_RUN_ID}" "${WITH_DIR}"
  aggregate_and_eval "${WITH_DIR}"
  generate_no_gold_input
  run_qwen "${NO_GOLD_RUN_ID}" "${NO_DIR}"
  aggregate_and_eval "${NO_DIR}"
  validate_outputs
  make_tables
  print_scores
  echo "[BOOTSTRAP REPRO] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee "${LOG_FILE}"
