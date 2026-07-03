#!/usr/bin/env bash
set -euo pipefail

BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
PYTHON_QWEN="${PYTHON_QWEN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/workspace/models/qwen30b}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_PROJECT}/exp/few_shot/runs/zh-dev-no-gold-rag-zhdoc-coder-all-mmr-oof-singleload}"
FOLDED_CSV="${FOLDED_CSV:-${BASE_PROJECT}/datasets/aligned_zh_folded.csv}"
DEV_GOLD="${DEV_GOLD:-${BASE_PROJECT}/datasets/mediqa-eval-2026-valid_1rater_zh.csv}"
DOCSTORE_JSON="${DOCSTORE_JSON:-${BASE_PROJECT}/RAG/storage_zh/docstore.json}"
CACHE_PREFIX="${CACHE_PREFIX:-${BASE_PROJECT}/RAG/storage_zh/cache/coder_all_cls}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-GanjinZero/coder_all}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-hf_cls_pool}"
SHOT_NUM="${SHOT_NUM:-20}"
BOOTSTRAP_NUM="${BOOTSTRAP_NUM:-1}"
SEED="${SEED:-114514}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
METRICS="${METRICS:-['factual-consistency-wgold','writing-style']}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${BASE_PROJECT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/fold_inputs" "${OUTPUT_DIR}/tmp"
LOG_FILE="${OUTPUT_DIR}/logs/run_$(date '+%Y%m%d_%H%M%S').log"

build_fold_csvs() {
  if [[ -s "${OUTPUT_DIR}/tmp/fold_0_shot.csv" && -s "${OUTPUT_DIR}/tmp/fold_0_infer.csv" ]]; then
    echo "[ZH-RAG-OOF] Reusing fold CSVs"
    return
  fi
  "${PYTHON_QWEN}" - "${FOLDED_CSV}" "${OUTPUT_DIR}/tmp" <<'PY'
import sys
from pathlib import Path
import pandas as pd

folded_csv = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(folded_csv)
df = df[df["lang"] == "zh"].copy()
for fold in sorted(df["fold"].dropna().astype(int).unique()):
    df[df["fold"].astype(int) != fold].to_csv(out_dir / f"fold_{fold}_shot.csv", index=False)
    df[df["fold"].astype(int) == fold].to_csv(out_dir / f"fold_{fold}_infer.csv", index=False)
PY
}

build_with_gold_input() {
  if [[ -s "${OUTPUT_DIR}/with_gold_input.json" ]]; then
    echo "[ZH-RAG-OOF] Reusing ${OUTPUT_DIR}/with_gold_input.json"
    return
  fi
  build_fold_csvs
  for fold in 0 1 2 3 4; do
    local fold_json="${OUTPUT_DIR}/fold_inputs/fold_${fold}.json"
    if [[ -s "${fold_json}" ]]; then
      echo "[ZH-RAG-OOF] Reusing ${fold_json}"
      continue
    fi
    "${PYTHON_QWEN}" "${BASE_PROJECT}/exp/few_shot/scripts/strategy/similarity_gold_shots.py" \
      --bs_sample_from "${OUTPUT_DIR}/tmp/fold_${fold}_shot.csv" \
      --infer_path "${OUTPUT_DIR}/tmp/fold_${fold}_infer.csv" \
      --output_path "${fold_json}" \
      --shot_num "${SHOT_NUM}" \
      --bootstrap_num "${BOOTSTRAP_NUM}" \
      --metrics "${METRICS}" \
      --embedding_model "${EMBEDDING_MODEL}" \
      --embedding_backend "${EMBEDDING_BACKEND}" \
      --strategy similarity_mmr \
      --en_only False \
      --seed "${SEED}"
  done
  "${PYTHON_QWEN}" - "${OUTPUT_DIR}/with_gold_input.json" "${OUTPUT_DIR}/fold_inputs"/fold_*.json <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
items = []
for path in sys.argv[2:]:
    items.extend(json.loads(Path(path).read_text(encoding="utf-8")))
out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved {len(items)} prompts to {out}")
PY
}

build_rag_input() {
  build_with_gold_input
  if [[ ! -s "${OUTPUT_DIR}/no_gold_input.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/paper_repro/make_no_gold_input.py" \
      --input-json "${OUTPUT_DIR}/with_gold_input.json" \
      --output-json "${OUTPUT_DIR}/no_gold_input.json" \
      --assert-no-gold
  else
    echo "[ZH-RAG-OOF] Reusing ${OUTPUT_DIR}/no_gold_input.json"
  fi

  if [[ ! -s "${OUTPUT_DIR}/input.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/rag_external/augment_rag_input_simple.py" \
      --input_json "${OUTPUT_DIR}/no_gold_input.json" \
      --output_json "${OUTPUT_DIR}/input.json" \
      --docstore_json "${DOCSTORE_JSON}" \
      --cache_prefix "${CACHE_PREFIX}" \
      --embed_model "${EMBEDDING_MODEL}" \
      --embedding_backend "${EMBEDDING_BACKEND}" \
      --device cuda \
      --retrieve_top_k 10 \
      --top_n 2
  else
    echo "[ZH-RAG-OOF] Reusing ${OUTPUT_DIR}/input.json"
  fi
}

run_infer_eval() {
  if [[ ! -s "${OUTPUT_DIR}/qwen.json" ]]; then
    "${PYTHON_QWEN}" "${BASE_PROJECT}/model_runners/infer_qwen3.py" \
      --model_path "${QWEN_MODEL}" \
      --data_path "${OUTPUT_DIR}/input.json" \
      --file_name "${OUTPUT_DIR}/qwen.json" \
      --run_id "zh-dev-no-gold-rag-zhdoc-coder-all-mmr-oof" \
      --device auto \
      --max_new_tokens "${MAX_NEW_TOKENS}"
  else
    echo "[ZH-RAG-OOF] Reusing ${OUTPUT_DIR}/qwen.json"
  fi

  "${PYTHON_QWEN}" "${BASE_PROJECT}/scripts/paper_repro/aggregate_gold_bootstrap_predictions.py" \
    --raw-json "${OUTPUT_DIR}/qwen.json" \
    --template-csv "${DEV_GOLD}" \
    --gold-csv "${DEV_GOLD}" \
    --eval-py "${BASE_PROJECT}/eval_zh.py" \
    --output-csv "${OUTPUT_DIR}/prediction.csv" \
    --all-runs-csv "${OUTPUT_DIR}/prediction_all_bootstraps.csv" \
    --score-json "${OUTPUT_DIR}/prediction.json" \
    --metrics "${METRICS}" \
    --lang zh
}

main() {
  cd "${BASE_PROJECT}"
  echo "[ZH-RAG-OOF] Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[ZH-RAG-OOF] OUTPUT_DIR=${OUTPUT_DIR}"
  build_rag_input
  run_infer_eval
  echo "[ZH-RAG-OOF] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main 2>&1 | tee -a "${LOG_FILE}"
