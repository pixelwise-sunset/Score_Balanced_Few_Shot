#!/usr/bin/env bash
set -euo pipefail

BASE_PROJECT="${BASE_PROJECT:-/workspace/MediQA}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/home/miniconda3/envs/qwen3/bin/python}"

export BASE_PROJECT

main() {
  cd "${BASE_PROJECT}"
  echo "[ZH-JBI] Started at $(date '+%Y-%m-%d %H:%M:%S')"

  "${BASE_PROJECT}/scripts/run_zh_bert_coder_all_dev_oof.sh"
  "${BASE_PROJECT}/scripts/run_zh_fewshot_bootstrap20x7_dev_oof.sh"
  "${BASE_PROJECT}/scripts/run_zh_rag_no_gold_dev_oof.sh"
  "${BASE_PROJECT}/scripts/run_zh_score_balanced_softmax20x7_dev_oof.sh"
  "${BASE_PROJECT}/scripts/run_zh_score_balanced_softmax20x7_test.sh"

  "${PYTHON_BIN}" "${BASE_PROJECT}/scripts/analysis/build_jbi_chinese_main_results.py"
  "${PYTHON_BIN}" "${BASE_PROJECT}/scripts/analysis/plot_jbi_chinese_main_table.py"

  echo "[ZH-JBI] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

main "$@"
