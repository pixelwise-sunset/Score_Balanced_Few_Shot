Few-Shot MEDIQA Pipeline
========================

This directory contains the few-shot judge pipeline ported from
`newchat111/MEDIQA-SUATBMI-ENSEMBLE`.

Main entry points:

- `scripts/make_shot_main.py`: builds image-aware few-shot prompts.
- `scripts/make_gold_shot_main.py`: builds text-only prompts with gold doctor responses.
- `scripts/strategy/bootstrap_shots.py`: repeats prompt construction with sampled shot sets.
- `scripts/strategy/bootstrap_gold_shots.py`: repeats gold-response prompt construction.
- `scripts/aggregate_bootstrap_predictions.py`: averages repeated bootstrap predictions.
- `scripts/prepare_few_shot_splits.py`: creates `train.csv` and `val.csv` from a folded aligned CSV.
- `model_runners/infer_qwen3.py`: runs Qwen3 text-only inference on prompt JSON files.
- `scripts/eval/eval_main.py`: converts model JSON outputs to MEDIQA CSV and scores them.

Expected data files are usually prepared under `exp/few_shot/datasets/`:

- `train.csv`: labeled examples for shots.
- `val.csv` or an aligned test/valid CSV: examples to score.

The aligned CSV should include at least:

`dataset, encounter_id, lang, candidate, candidate_author_id, metric, label, query_text, image_path, gold_texts`.

For the paper's gold-response Qwen setting, start from
`exp_scripts/run_qwen_gold_bootstrap_local.sh` and update `QWEN_MODEL` before running.
