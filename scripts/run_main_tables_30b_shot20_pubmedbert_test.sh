#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_QWEN="${PYTHON_QWEN:-/data/kunfeng/miniconda3/envs/qwen3/bin/python}"
PYTHON_BERT="${PYTHON_BERT:-/data/liyuan/conda_envs/yolo/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_tables_30b_shot20_pubmedbert_test}"
ASSET_DIR="${ASSET_DIR:-datasets/test_assets}"
TEST_GOLD="${TEST_GOLD:-${ASSET_DIR}/test_gold_en.csv}"
TEST_TEMPLATE="${TEST_TEMPLATE:-${ASSET_DIR}/test_template_en.csv}"
TEST_ALIGNED="${TEST_ALIGNED:-${ASSET_DIR}/test_aligned_en.csv}"
SHOT_POOL="${SHOT_POOL:-datasets/aligned_en_folded.csv}"
RAG_PERSIST_DIR="${RAG_PERSIST_DIR:-RAG/storage_pubmedbert}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/public_models/bert/pubmedbert}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-hf_mean_pool}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
QWEN_MODEL="${QWEN_MODEL:-/data/public_models/Qwen3-30B-A3B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']}"

WITH_GOLD_RUN_ID="${WITH_GOLD_RUN_ID:-main30b-test-with-gold-pubmedbert-mmr-singleload}"
NO_GOLD_RUN_ID="${NO_GOLD_RUN_ID:-main30b-test-no-gold-pubmedbert-mmr-singleload}"
RAG_RUN_ID="${RAG_RUN_ID:-main30b-test-no-gold-rag-v2-pubmedbert-query-mmr-singleload}"

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/run_test_tables_$(date +%Y%m%d_%H%M%S).log"

run_bert() {
  local name="$1"
  local strategy="$2"
  local bs="$3"
  local out_dir="${OUTPUT_ROOT}/bert_${name}"
  if [[ -f "${out_dir}/prediction.json" ]]; then
    echo "[TEST TABLES] BERT ${name} exists; skipping"
    return
  fi
  "${PYTHON_BERT}" scripts/bert/train_predict_full_dev_bert.py \
    --strategy_type "${strategy}" \
    --train_csv "${SHOT_POOL}" \
    --test_aligned_csv "${TEST_ALIGNED}" \
    --test_template_csv "${TEST_TEMPLATE}" \
    --test_gold_csv "${TEST_GOLD}" \
    --output_dir "${out_dir}" \
    --model_name "/data/public_models/bert/pubmedbert" \
    --epochs 10 \
    --batch_size "${bs}" \
    --lr 2e-5
}

run_with_gold_fewshot() {
  PYTHON_BIN="${PYTHON_QWEN}" \
  QWEN_MODEL="${QWEN_MODEL}" \
  RUN_ID="${WITH_GOLD_RUN_ID}" \
  SHOT_POOL="${SHOT_POOL}" \
  INFER_PATH="${TEST_ALIGNED}" \
  TEST_GOLD="${TEST_GOLD}" \
  TEST_TEMPLATE="${TEST_TEMPLATE}" \
  EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
  EMBEDDING_BACKEND="${EMBEDDING_BACKEND}" \
  EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  METRICS="${METRICS}" \
  bash scripts/fewshot_test_singleload.sh
}

run_no_gold_fewshot() {
  local with_dir="exp/few_shot/runs/${WITH_GOLD_RUN_ID}"
  local no_dir="exp/few_shot/runs/${NO_GOLD_RUN_ID}"
  mkdir -p "${no_dir}/logs"
  "${PYTHON_QWEN}" exp/few_shot/scripts/make_no_gold_input.py \
    --input_json "${with_dir}/input.json" \
    --output_json "${no_dir}/input.json" \
    --assert_no_gold
  "${PYTHON_QWEN}" model_runners/infer_qwen3.py \
    --model_path "${QWEN_MODEL}" \
    --data_path "${no_dir}/input.json" \
    --file_name "${no_dir}/qwen.json" \
    --run_id "${NO_GOLD_RUN_ID}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  "${PYTHON_QWEN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
    --prediction_path "${no_dir}/qwen.json" \
    --true_path "${TEST_TEMPLATE}" \
    --output_csv "${no_dir}/prediction.csv" \
    --score_json "${no_dir}/aggregate_score.json" \
    --metrics "${METRICS}" \
    --markdown
  "${PYTHON_QWEN}" eval_en.py "${TEST_GOLD}" "${no_dir}/prediction.csv" "${no_dir}/prediction.json"
}

run_rag() {
  local no_dir="exp/few_shot/runs/${NO_GOLD_RUN_ID}"
  local rag_dir="exp/few_shot/runs/${RAG_RUN_ID}"
  mkdir -p "${rag_dir}/logs"
  "${PYTHON_QWEN}" scripts/rag_external/augment_rag_input.py \
    --input_json "${no_dir}/input.json" \
    --output_json "${rag_dir}/input.json" \
    --persist_dir "${RAG_PERSIST_DIR}" \
    --embed_model "${EMBEDDING_MODEL}" \
    --embedding_backend "${EMBEDDING_BACKEND}" \
    --embedding_max_length "${EMBEDDING_MAX_LENGTH}" \
    --query_source target_query \
    --insert_location after_llm_response \
    --retrieve_top_k 10 \
    --rerank_top_n 2 \
    --max_evidence_chars 1200 \
    --include_metadata \
    --disable_rerank
  "${PYTHON_QWEN}" model_runners/infer_qwen3.py \
    --model_path "${QWEN_MODEL}" \
    --data_path "${rag_dir}/input.json" \
    --file_name "${rag_dir}/qwen.json" \
    --run_id "${RAG_RUN_ID}" \
    --device auto \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  "${PYTHON_QWEN}" exp/few_shot/scripts/aggregate_bootstrap_predictions.py \
    --prediction_path "${rag_dir}/qwen.json" \
    --true_path "${TEST_TEMPLATE}" \
    --output_csv "${rag_dir}/prediction.csv" \
    --score_json "${rag_dir}/aggregate_score.json" \
    --metrics "${METRICS}" \
    --markdown
  "${PYTHON_QWEN}" eval_en.py "${TEST_GOLD}" "${rag_dir}/prediction.csv" "${rag_dir}/prediction.json"
}

make_ensembles_and_tables() {
  local fewshot_with="exp/few_shot/runs/${WITH_GOLD_RUN_ID}"
  local fewshot_no="exp/few_shot/runs/${NO_GOLD_RUN_ID}"
  local rag="exp/few_shot/runs/${RAG_RUN_ID}"

  for kind in matched historical; do
    local bert_with="${OUTPUT_ROOT}/bert_${kind}_with_gold"
    local bert_no="${OUTPUT_ROOT}/bert_${kind}_without_gold"
    local ens_with="${OUTPUT_ROOT}/ensemble_${kind}_metricwise_with_gold"
    local ens_no="${OUTPUT_ROOT}/ensemble_${kind}_metricwise_without_gold"
    mkdir -p "${ens_with}" "${ens_no}"

    "${PYTHON_QWEN}" scripts/select_metricwise_predictions.py \
      --source "Few-shot=${fewshot_with}/prediction.csv" \
      --source "BERT=${bert_with}/prediction.csv" \
      --metric_source "disagree_flag=Few-shot" \
      --metric_source "completeness=BERT" \
      --metric_source "factual-accuracy=Few-shot" \
      --metric_source "relevance=BERT" \
      --metric_source "writing-style=BERT" \
      --metric_source "overall=Few-shot" \
      --output_csv "${ens_with}/prediction.csv" \
      --source_map_json "${ens_with}/source_map.json" \
      --rater_id "bert_fewshot_${kind}_metricwise"
    "${PYTHON_QWEN}" eval_en.py "${TEST_GOLD}" "${ens_with}/prediction.csv" "${ens_with}/prediction.json"

    "${PYTHON_QWEN}" scripts/select_metricwise_predictions.py \
      --source "Few-shot=${fewshot_no}/prediction.csv" \
      --source "BERT=${bert_no}/prediction.csv" \
      --source "Few-shot w/ RAG=${rag}/prediction.csv" \
      --metric_source "disagree_flag=Few-shot w/ RAG" \
      --metric_source "completeness=Few-shot w/ RAG" \
      --metric_source "factual-accuracy=Few-shot" \
      --metric_source "relevance=Few-shot w/ RAG" \
      --metric_source "writing-style=Few-shot w/ RAG" \
      --metric_source "overall=Few-shot" \
      --output_csv "${ens_no}/prediction.csv" \
      --source_map_json "${ens_no}/source_map.json" \
      --rater_id "bert_fewshot_rag_${kind}_metricwise"
    "${PYTHON_QWEN}" eval_en.py "${TEST_GOLD}" "${ens_no}/prediction.csv" "${ens_no}/prediction.json"

    "${PYTHON_QWEN}" scripts/summarize_main_tables.py \
      --entry "Few-shot=${fewshot_with}/prediction.json" \
      --entry "BERT=${bert_with}/prediction.json" \
      --entry "Metric-wise Ensemble=${ens_with}/prediction.json" \
      --output_csv "${OUTPUT_ROOT}/table_with_gold_${kind}.csv"
    "${PYTHON_QWEN}" scripts/summarize_main_tables.py \
      --entry "Few-shot=${fewshot_no}/prediction.json" \
      --entry "BERT=${bert_no}/prediction.json" \
      --entry "Few-shot w/ RAG=${rag}/prediction.json" \
      --entry "Metric-wise Ensemble=${ens_no}/prediction.json" \
      --output_csv "${OUTPUT_ROOT}/table_without_gold_${kind}.csv"
  done
}

run_checks() {
  "${PYTHON_QWEN}" - <<PY
import json
import pandas as pd
from pathlib import Path
test_gold = pd.read_csv("${TEST_GOLD}")
template = pd.read_csv("${TEST_TEMPLATE}")
aligned = pd.read_csv("${TEST_ALIGNED}")
print("[CHECK] test_gold rows", len(test_gold))
print("[CHECK] template rows", len(template))
print("[CHECK] aligned rows", len(aligned))
assert len(test_gold) == 6948
assert len(template) == 3474
assert len(aligned) == 3474
for p in [
  "${OUTPUT_ROOT}/table_with_gold_matched.csv",
  "${OUTPUT_ROOT}/table_without_gold_matched.csv",
  "${OUTPUT_ROOT}/table_with_gold_historical.csv",
  "${OUTPUT_ROOT}/table_without_gold_historical.csv",
]:
    assert Path(p).exists(), p
no_gold = json.load(open("exp/few_shot/runs/${NO_GOLD_RUN_ID}/input.json"))
assert "gold responses" not in json.dumps(no_gold, ensure_ascii=False).lower()
rag = json.load(open("exp/few_shot/runs/${RAG_RUN_ID}/input.json"))
assert "[Relevant Clinical Guidelines]" in json.dumps(rag, ensure_ascii=False)
print("[CHECK] test tables checks passed")
PY
}

main() {
  echo "[TEST TABLES] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[TEST TABLES] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[TEST TABLES] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  run_bert matched_with_gold matched_with_gold 16
  run_bert matched_without_gold matched_without_gold 16
  run_bert historical_with_gold historical_dataset_specific 16
  run_bert historical_without_gold historical_no_image 32
  run_with_gold_fewshot
  run_no_gold_fewshot
  run_rag
  make_ensembles_and_tables
  run_checks
  echo "[TEST TABLES] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee "${LOG_FILE}"
