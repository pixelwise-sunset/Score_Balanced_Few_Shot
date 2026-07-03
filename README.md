# MediQA JBI Main-Table Reproduction Code

This repository contains the code needed to reproduce the English and Chinese
main-result tables for the JBI manuscript. It is scoped to the main experiments
only: BERT baselines, random/bootstrap few-shot, RAG few-shot, score-balanced
few-shot, normalized main-result builders, and main table plotting.

It does not include datasets, manuscript-only analyses, paper drafts, generated
predictions, raw LLM outputs, model checkpoints, or RAG indexes.

## Data

Datasets are intentionally not included in this code repository. The experiment
CSV files are distributed as a separate dataset archive. To reproduce the
experiments, extract that archive at the repository root so that it creates:

```text
datasets/
```

The archive contains the official MEDIQA-EVAL 2026 development/test CSVs and
the aligned English/Chinese CSVs used by the scripts. Generated predictions and
model outputs are not part of the dataset archive.

## External Runtime Assets

Full reruns require model/runtime assets that are not included:

- Qwen3-30B-A3B or an equivalent local model path for LLM judging.
- BERT encoder checkpoints or Hugging Face cache access.
- RAG docstores/indexes for the RAG baselines.
- A CUDA environment for BERT training and Qwen inference.

The scripts are written so that finished prediction directories can be reused.
If the expected `results/.../prediction.csv` files already exist, the main-table
builders can be run without rerunning GPU inference.

## Environment

Install Python dependencies in a Python 3.10+ environment:

```bash
pip install -r requirements.txt
```

For the original GPU setup, use `environment.yaml` or
`requirements_fewshot.txt`. The School-server defaults can be overridden:

```bash
export BASE_PROJECT=/path/to/MediQA_journal_code_release
export PYTHON_QWEN=/path/to/qwen/env/bin/python
export PYTHON_BIN=/path/to/python
export QWEN_MODEL=/path/to/Qwen3-30B-A3B
export CUDA_VISIBLE_DEVICES=0,1
```

## English Main Table

From the repository root:

```bash
# BERT baselines
bash scripts/bert/run_retrain_multitask_with_without_gold.sh

# Random/bootstrap few-shot dev OOF and test aggregation
bash scripts/run_en_dev_selected_bootstrap_pipeline.sh
bash scripts/run_competition_bootstrap_qwen30b_test.sh

# Score-balanced few-shot dev OOF and test
bash scripts/run_en_score_balanced_softmax20x7_dev_oof.sh
bash scripts/run_en_shot_selection_probabilistic20x7.sh

# Reference-free RAG baseline
bash scripts/run_main_tables_30b_shot20_pubmedbert.sh
bash scripts/run_main_tables_30b_shot20_pubmedbert_test.sh

# Normalize sources, select metric-wise ensemble on dev, and draw the main figure
python scripts/analysis/build_jbi_main_results.py
python scripts/analysis/plot_jbi_english_main_table.py
```

The normalized English outputs are written under
`results/jbi_main_results_20260618/`.

## English Coverage-Selection Ablations

These optional ablations are separate from the main tables. They compare random
few-shot, global hard metric-bin coverage, per-sample coverage+similarity, and
score-balanced softmax:

```bash
# Dev OOF ablations
bash scripts/run_en_metric_bin_coverage20x7_dev_oof.sh
bash scripts/run_en_per_sample_coverage_similarity20x7_dev_oof.sh

# Test ablations
bash scripts/run_en_coverage_selection20x7_test.sh

# Build ablation tables and figure
python scripts/analysis/build_coverage_selection_ablation.py
```

The ablation outputs are written under
`results/jbi_main_results_20260618/coverage_selection_ablation/`.

## Chinese Main Table

From the repository root:

```bash
# BERT baselines
bash scripts/run_zh_bert_coder_all_dev_oof.sh
bash scripts/run_zh_bert_coder_all_test.sh

# Random/bootstrap, RAG, and score-balanced few-shot
bash scripts/run_zh_fewshot_bootstrap20x7_dev_oof.sh
bash scripts/run_competition_bootstrap_qwen30b_zh_test.sh
bash scripts/run_zh_rag_no_gold_dev_oof.sh
bash scripts/run_zh_score_balanced_softmax20x7_dev_oof.sh
bash scripts/run_zh_score_balanced_softmax20x7_test.sh

# Normalize sources, select metric-wise ensemble on dev, and draw the main figure
python scripts/analysis/build_jbi_chinese_main_results.py
python scripts/analysis/plot_jbi_chinese_main_table.py
```

The convenience wrapper below runs the Chinese sequence except the separate
BERT test helper if those outputs already exist:

```bash
bash scripts/run_zh_jbi_main_results_pipeline.sh
```

The normalized Chinese outputs are written under
`results/jbi_main_results_20260618/zh/`.

## Quick Non-GPU Check

To verify the release layout without running inference:

```bash
python -m py_compile \
  eval_en.py eval_zh.py \
  scripts/analysis/build_jbi_main_results.py \
  scripts/analysis/build_jbi_chinese_main_results.py \
  scripts/analysis/plot_jbi_english_main_table.py \
  scripts/analysis/plot_jbi_chinese_main_table.py
```

The builders will still require the expected prediction CSVs under `results/`
unless the full experiment sequence has already been run.
