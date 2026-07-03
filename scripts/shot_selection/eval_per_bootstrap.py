import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.eval.eval_json import en_small_sample_merge, get_prediction
from utils.eval.mediqa_eval_script import EVAL_COLS_UNIQUE, get_correlations


METRIC_KEYS = [
    ("ALL", "ALL_en_ALL_mean"),
    ("disagree_flag", "disagree_flag"),
    ("completeness", "completeness"),
    ("factual-accuracy", "factual-accuracy"),
    ("relevance", "relevance"),
    ("writing-style", "writing-style"),
    ("overall", "overall"),
]


def item_bootstrap_id(item: dict) -> int:
    return int(item["input"][0]["key"][0]["bootstrap_id"])


def score_predictions(true_df: pd.DataFrame, pred_df: pd.DataFrame, metrics: list[str]) -> dict:
    merged_df = en_small_sample_merge(true_df=true_df, pred_df=pred_df)
    scores = {}
    total_score = 0.0
    for metric in metrics:
        per_metric_df = merged_df[merged_df["metric"] == metric]
        kendalltau, pearson, spearman, _, _, _ = get_correlations(
            x=per_metric_df["value_x"],
            y=per_metric_df["value_y"],
        )
        mean_corr = (kendalltau + pearson + spearman) / 3
        scores[metric] = mean_corr
        total_score += mean_corr
    scores["ALL_en_ALL_mean"] = total_score / len(metrics)
    return scores


def aggregate_raw(raw_path: Path, true_path: str, metrics: list[str], markdown: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    true_df, pred_df = get_prediction(
        true_path=true_path,
        prediction_path=str(raw_path),
        in_mark_down=markdown,
        metrics=metrics,
    )
    value_cols = EVAL_COLS_UNIQUE + ["rater_id"]
    averaged = (
        pred_df.groupby(value_cols, as_index=False, dropna=False)["value"]
        .mean()
        .loc[:, pred_df.columns]
    )
    scores = score_predictions(true_df=true_df, pred_df=averaged, metrics=metrics)
    return true_df, averaged, scores


def write_score(path: Path, scores: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True)
    parser.add_argument("--true_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    metrics = ast.literal_eval(args.metrics)
    raw_items = json.loads(Path(args.prediction_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_ids = sorted({item_bootstrap_id(item) for item in raw_items})
    rows = []
    for bootstrap_id in bootstrap_ids:
        subset = [item for item in raw_items if item_bootstrap_id(item) == bootstrap_id]
        boot_dir = output_dir / f"bootstrap_{bootstrap_id}"
        boot_dir.mkdir(parents=True, exist_ok=True)
        raw_path = boot_dir / "raw.json"
        raw_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
        _, averaged, scores = aggregate_raw(
            raw_path=raw_path,
            true_path=args.true_path,
            metrics=metrics,
            markdown=args.markdown,
        )
        averaged.to_csv(boot_dir / "prediction.csv", index=False)
        write_score(boot_dir / "prediction.json", scores)
        row = {
            "context": args.context,
            "method": args.method,
            "bootstrap_id": bootstrap_id,
            "prediction_json": str(boot_dir / "prediction.json"),
            "prediction_csv": str(boot_dir / "prediction.csv"),
        }
        row.update({name: scores.get(key) for name, key in METRIC_KEYS})
        rows.append(row)

    summary_path = output_dir / "per_bootstrap_summary.csv"
    fieldnames = ["context", "method", "bootstrap_id", "prediction_json", "prediction_csv"] + [
        name for name, _ in METRIC_KEYS
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} per-bootstrap rows to {summary_path}")


if __name__ == "__main__":
    main()
