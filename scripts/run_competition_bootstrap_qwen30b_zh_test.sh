#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_QWEN="${PYTHON_QWEN:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/competition_bootstrap_qwen30b_test}"
ASSET_DIR="${ASSET_DIR:-datasets/test_assets}"
SHOT_POOL="${SHOT_POOL:-datasets/aligned_zh_folded.csv}"
TEST_ALIGNED="${TEST_ALIGNED:-${ASSET_DIR}/test_aligned_zh.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-${ASSET_DIR}/test_template_zh.csv}"
TEST_GOLD="${TEST_GOLD:-${ASSET_DIR}/test_gold_zh.csv}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-7}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['factual-consistency-wgold','writing-style']}"

WITH_GOLD_RUN_ID="${WITH_GOLD_RUN_ID:-competition-qwen30b-bootstrap20x7-test-zh-with-gold}"
NO_GOLD_RUN_ID="${NO_GOLD_RUN_ID:-competition-qwen30b-bootstrap20x7-test-zh-no-gold}"

WITH_DIR="exp/few_shot/runs/${WITH_GOLD_RUN_ID}"
NO_DIR="exp/few_shot/runs/${NO_GOLD_RUN_ID}"
mkdir -p "${OUTPUT_ROOT}/logs" "${WITH_DIR}/logs" "${NO_DIR}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/run_zh_$(date +%Y%m%d_%H%M%S).log"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ZH BOOTSTRAP] Missing required file: ${path}" >&2
    exit 1
  fi
}

check_assets() {
  require_file "${SHOT_POOL}"
  require_file "${TEST_ALIGNED}"
  require_file "${TEST_TEMPLATE}"
  require_file "${TEST_GOLD}"
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
    print(f"[ZH BOOTSTRAP] {name}: {path} rows={len(df)}")
assert len(pd.read_csv("${SHOT_POOL}")) == 966
assert len(pd.read_csv("${TEST_GOLD}")) == 1158
assert len(pd.read_csv("${TEST_TEMPLATE}")) == 1158
assert len(pd.read_csv("${TEST_ALIGNED}")) == 1158
PY
}

build_assets_if_needed() {
  if [[ -s "${TEST_ALIGNED}" && -s "${TEST_TEMPLATE}" && -s "${TEST_GOLD}" ]]; then
    echo "[ZH BOOTSTRAP] Chinese test assets exist; skipping build"
    return
  fi
  "${PYTHON_QWEN}" scripts/testset/build_chinese_test_assets.py \
    --test_gold datasets/mediqa-eval-2026-test.csv \
    --output_dir "${ASSET_DIR}"
}

generate_with_gold_input() {
  local input_json="${WITH_DIR}/input.json"
  if [[ -s "${input_json}" ]]; then
    echo "[ZH BOOTSTRAP] with-gold input exists; skipping: ${input_json}"
    return
  fi
  "${PYTHON_QWEN}" exp/few_shot/scripts/strategy/bootstrap_gold_shots.py \
    --bs_sample_from "${SHOT_POOL}" \
    --infer_path "${TEST_ALIGNED}" \
    --output_path "${input_json}" \
    --shot_num "${SHOT_NUM}" \
    --bootstrap_num "${BOOTSTRAP_NUM}" \
    --metrics "${METRICS}" \
    --en_only False \
    --sample_with_replacement True \
    --seed "${SEED}"
}

generate_no_gold_input() {
  local input_json="${NO_DIR}/input.json"
  if [[ -s "${input_json}" ]]; then
    echo "[ZH BOOTSTRAP] no-gold input exists; skipping: ${input_json}"
    return
  fi
  "${PYTHON_QWEN}" exp/few_shot/scripts/make_no_gold_input.py \
    --input_json "${WITH_DIR}/input.json" \
    --output_json "${input_json}" \
    --assert_no_gold
}

run_qwen() {
  local run_id="$1"
  local run_dir="$2"
  local input_json="${run_dir}/input.json"
  local raw_json="${run_dir}/qwen30b.json"
  if [[ -s "${raw_json}" ]]; then
    echo "[ZH BOOTSTRAP] raw Qwen output exists; skipping: ${raw_json}"
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
    echo "[ZH BOOTSTRAP] aggregate output exists; skipping: ${pred_csv}"
  fi
  if [[ ! -s "${score_json}" ]]; then
    "${PYTHON_QWEN}" eval_zh.py "${TEST_GOLD}" "${pred_csv}" "${score_json}"
  else
    echo "[ZH BOOTSTRAP] eval output exists; skipping: ${score_json}"
  fi
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
    assert len(pred) == 1158, (name, len(pred))
    assert set(pred["metric"].unique()) == {"factual-consistency-wgold", "writing-style"}, name
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    assert "ALL-zh-ALL-mean" in scores, name
    print(f"[ZH BOOTSTRAP] {name} ALL-zh-ALL-mean={scores['ALL-zh-ALL-mean']}")
PY
}

make_table() {
  "${PYTHON_QWEN}" - <<PY
import json
import pandas as pd

entries = [
    ("Bootstrap Few-shot w/ gold", "${WITH_DIR}/prediction.json"),
    ("Bootstrap Few-shot w/o gold", "${NO_DIR}/prediction.json"),
]
rows = []
for method, path in entries:
    with open(path, encoding="utf-8") as f:
        scores = json.load(f)
    rows.append({
        "method": method,
        "score_json": path,
        "all_zh_all_mean": scores.get("ALL-zh-ALL-mean"),
        "factual_consistency_wgold_mean": scores.get("ALL-zh-factual-consistency-wgold-mean"),
        "writing_style_mean": scores.get("ALL-zh-writing-style-mean"),
    })
out = "${OUTPUT_ROOT}/table_zh_bootstrap_reproduction.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"[ZH BOOTSTRAP] Saved {out}")
print(pd.DataFrame(rows).to_string(index=False))
PY
}

main() {
  {
    echo "[ZH BOOTSTRAP] Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[ZH BOOTSTRAP] OUTPUT_ROOT=${OUTPUT_ROOT}"
    echo "[ZH BOOTSTRAP] QWEN_MODEL=${QWEN_MODEL}"
    echo "[ZH BOOTSTRAP] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[ZH BOOTSTRAP] SHOT_NUM=${SHOT_NUM} BOOTSTRAP_NUM=${BOOTSTRAP_NUM} SEED=${SEED}"
    build_assets_if_needed
    check_assets
    generate_with_gold_input
    run_qwen "${WITH_GOLD_RUN_ID}" "${WITH_DIR}"
    aggregate_and_eval "${WITH_DIR}"
    generate_no_gold_input
    run_qwen "${NO_GOLD_RUN_ID}" "${NO_DIR}"
    aggregate_and_eval "${NO_DIR}"
    validate_outputs
    make_table
    echo "[ZH BOOTSTRAP] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee "${LOG_FILE}"
}

main "$@"
