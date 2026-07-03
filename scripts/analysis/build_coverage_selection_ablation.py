#!/usr/bin/env python3
"""Build English coverage-selection ablation tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_en import EVAL_COLS_UNIQUE, score_correlations


KEY_COLS = ["dataset", "encounter_id", "lang", "candidate", "candidate_author_id", "metric"]
OUT_COLS = KEY_COLS + ["rater_id", "value"]
METRICS = ["disagree_flag", "completeness", "factual-accuracy", "relevance", "writing-style", "overall"]
SCORE_KEYS = [
    ("all_en_all_mean", "ALL-en-ALL-mean"),
    ("disagree_flag", "ALL-en-disagree_flag-mean"),
    ("completeness", "ALL-en-completeness-mean"),
    ("factual_accuracy", "ALL-en-factual-accuracy-mean"),
    ("relevance", "ALL-en-relevance-mean"),
    ("writing_style", "ALL-en-writing-style-mean"),
    ("overall", "ALL-en-overall-mean"),
]
EXPECTED_ROWS = {"dev": 2898, "test": 3474}
EXPECTED_PER_METRIC = {"dev": 483, "test": 579}


@dataclass(frozen=True)
class SourceSpec:
    context: str
    split: str
    method_key: str
    label: str
    prediction_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/jbi_main_results_20260618/coverage_selection_ablation")
    parser.add_argument("--dev-gold-csv", default="datasets/mediqa-eval-2026-valid_1rater_en.csv")
    parser.add_argument("--test-gold-csv", default="datasets/test_assets/test_gold_en.csv")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def default_specs() -> list[SourceSpec]:
    random_dev = Path("results/fewshot_bootstrap20x7_original_20260617")
    random_test_with = Path("exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-with-gold")
    random_test_no = Path("exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-no-gold")
    score_dev = Path("results/en_shot_selection_probabilistic20x7/dev_oof/score_balanced_softmax20x7")
    score_test = Path("results/en_shot_selection_probabilistic20x7/runs/score_balanced_softmax20x7")
    coverage_dev = Path("results/en_shot_selection_probabilistic20x7/dev_oof/metric_bin_coverage20x7")
    coverage_test = Path("results/en_shot_selection_probabilistic20x7/runs/metric_bin_coverage20x7")
    hybrid_dev = Path("results/en_shot_selection_probabilistic20x7/dev_oof/per_sample_coverage_similarity20x7")
    hybrid_test = Path("results/en_shot_selection_probabilistic20x7/runs/per_sample_coverage_similarity20x7")

    return [
        SourceSpec("with_gold", "dev", "random_fewshot20x7", "Random few-shot 20x7", random_dev / "with_gold_dev_oof_20x7/prediction.csv"),
        SourceSpec("with_gold", "test", "random_fewshot20x7", "Random few-shot 20x7", random_test_with / "prediction.csv"),
        SourceSpec("without_gold", "dev", "random_fewshot20x7", "Random few-shot 20x7", random_dev / "without_gold_dev_oof_20x7/prediction.csv"),
        SourceSpec("without_gold", "test", "random_fewshot20x7", "Random few-shot 20x7", random_test_no / "prediction.csv"),
        SourceSpec("with_gold", "dev", "metric_bin_coverage20x7", "Global metric-bin coverage 20x7", coverage_dev / "with_gold/prediction.csv"),
        SourceSpec("with_gold", "test", "metric_bin_coverage20x7", "Global metric-bin coverage 20x7", coverage_test / "with_gold/prediction.csv"),
        SourceSpec("without_gold", "dev", "metric_bin_coverage20x7", "Global metric-bin coverage 20x7", coverage_dev / "no_gold/prediction.csv"),
        SourceSpec("without_gold", "test", "metric_bin_coverage20x7", "Global metric-bin coverage 20x7", coverage_test / "no_gold/prediction.csv"),
        SourceSpec("with_gold", "dev", "per_sample_coverage_similarity20x7", "Per-sample coverage+similarity 20x7", hybrid_dev / "with_gold/prediction.csv"),
        SourceSpec("with_gold", "test", "per_sample_coverage_similarity20x7", "Per-sample coverage+similarity 20x7", hybrid_test / "with_gold/prediction.csv"),
        SourceSpec("without_gold", "dev", "per_sample_coverage_similarity20x7", "Per-sample coverage+similarity 20x7", hybrid_dev / "no_gold/prediction.csv"),
        SourceSpec("without_gold", "test", "per_sample_coverage_similarity20x7", "Per-sample coverage+similarity 20x7", hybrid_test / "no_gold/prediction.csv"),
        SourceSpec("with_gold", "dev", "score_balanced_softmax20x7", "Score-balanced softmax 20x7", score_dev / "with_gold/prediction.csv"),
        SourceSpec("with_gold", "test", "score_balanced_softmax20x7", "Score-balanced softmax 20x7", score_test / "with_gold/prediction.csv"),
        SourceSpec("without_gold", "dev", "score_balanced_softmax20x7", "Score-balanced softmax 20x7", score_dev / "no_gold/prediction.csv"),
        SourceSpec("without_gold", "test", "score_balanced_softmax20x7", "Score-balanced softmax 20x7", score_test / "no_gold/prediction.csv"),
    ]


def load_prediction(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in OUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.loc[:, OUT_COLS].copy()
    conflicts = df.groupby(KEY_COLS, dropna=False)["value"].nunique().reset_index()
    bad = conflicts[conflicts["value"] > 1]
    if not bad.empty:
        raise ValueError(f"{path} has {len(bad)} duplicate keys with conflicting values")
    df = df.drop_duplicates(KEY_COLS, keep="first").sort_values(KEY_COLS).reset_index(drop=True)
    if len(df) != EXPECTED_ROWS[split]:
        raise ValueError(f"{path} normalized to {len(df)} rows; expected {EXPECTED_ROWS[split]}")
    counts = df["metric"].value_counts().to_dict()
    for metric in METRICS:
        if counts.get(metric, 0) != EXPECTED_PER_METRIC[split]:
            raise ValueError(f"{path} has {counts.get(metric, 0)} rows for {metric}; expected {EXPECTED_PER_METRIC[split]}")
    return df


def evaluate(gold_csv: Path, pred_df: pd.DataFrame, score_json: Path) -> dict:
    scores = score_correlations(pd.read_csv(gold_csv), pred_df.drop_duplicates(subset=EVAL_COLS_UNIQUE))
    score_json.parent.mkdir(parents=True, exist_ok=True)
    score_json.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def normalize_and_score(spec: SourceSpec, out_root: Path, dev_gold: Path, test_gold: Path) -> dict:
    out_dir = out_root / "normalized" / spec.context / spec.split / spec.method_key
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = load_prediction(spec.prediction_csv, spec.split)
    pred_path = out_dir / "prediction.csv"
    score_path = out_dir / "prediction.json"
    pred.to_csv(pred_path, index=False)
    scores = evaluate(dev_gold if spec.split == "dev" else test_gold, pred, score_path)
    provenance = {
        "context": spec.context,
        "split": spec.split,
        "method_key": spec.method_key,
        "method_label": spec.label,
        "source_prediction_csv": str(spec.prediction_csv),
        "normalized_prediction_csv": str(pred_path),
        "normalized_score_json": str(score_path),
        "rows": len(pred),
        "metric_counts": pred["metric"].value_counts().sort_index().to_dict(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    row = {
        "context": spec.context,
        "split": spec.split,
        "method_key": spec.method_key,
        "method": spec.label,
        "prediction_csv": str(pred_path),
        "score_json": str(score_path),
    }
    for col, score_key in SCORE_KEYS:
        row[col] = scores.get(score_key)
    return row


def write_summary(rows: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "coverage_selection_ablation_scores.csv"
    json_path = out_root / "coverage_selection_ablation_scores.json"
    fieldnames = [
        "context",
        "split",
        "method_key",
        "method",
        "prediction_csv",
        "score_json",
    ] + [col for col, _ in SCORE_KEYS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


def plot_overall(rows: list[dict], out_root: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    method_order = [
        "random_fewshot20x7",
        "metric_bin_coverage20x7",
        "per_sample_coverage_similarity20x7",
        "score_balanced_softmax20x7",
    ]
    method_labels = {
        row["method_key"]: row["method"]
        for row in rows
    }
    panels = [
        ("with_gold", "dev", "With Gold - Dev"),
        ("with_gold", "test", "With Gold - Test"),
        ("without_gold", "dev", "Without Gold - Dev"),
        ("without_gold", "test", "Without Gold - Test"),
    ]
    colors = ["#8FA7C2", "#E0B75A", "#69A87D", "#C96B5B"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True)
    for ax, (context, split, title) in zip(axes.flat, panels):
        panel = df[(df["context"] == context) & (df["split"] == split)].copy()
        values = []
        labels = []
        bar_colors = []
        for idx, method_key in enumerate(method_order):
            match = panel[panel["method_key"] == method_key]
            if match.empty:
                continue
            values.append(float(match["all_en_all_mean"].iloc[0]))
            labels.append(method_labels.get(method_key, method_key))
            bar_colors.append(colors[idx])
        ax.bar(range(len(values)), values, color=bar_colors, edgecolor="#334155", linewidth=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("ALL-en-ALL-mean")
        ax.grid(axis="y", alpha=0.25)
        for pos, value in enumerate(values):
            ax.text(pos, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Coverage Selection Ablation", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / "coverage_selection_ablation_overall.png"
    pdf = fig_dir / "coverage_selection_ablation_overall.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    dev_gold = Path(args.dev_gold_csv)
    test_gold = Path(args.test_gold_csv)
    rows = []
    missing = []
    for spec in default_specs():
        if not spec.prediction_csv.exists():
            missing.append(str(spec.prediction_csv))
            continue
        rows.append(normalize_and_score(spec, out_root, dev_gold, test_gold))

    if missing:
        manifest = {
            "missing_sources": missing,
            "completed_rows": len(rows),
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "missing_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        message = f"{len(missing)} source prediction files are missing; wrote missing_sources.json"
        if not args.allow_missing:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")

    write_summary(rows, out_root)
    plot_overall(rows, out_root)


if __name__ == "__main__":
    main()
