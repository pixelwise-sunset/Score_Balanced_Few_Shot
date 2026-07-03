import argparse
import ast
import json
import os

import pandas as pd

from utils.eval.eval_json import en_small_sample_merge, get_prediction
from utils.eval.mediqa_eval_script import EVAL_COLS_UNIQUE, get_correlations


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True)
    parser.add_argument("--true_path", default="datasets/mediqa-eval-2026-valid.csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--score_json", default="")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    metrics = ast.literal_eval(args.metrics)
    true_df, pred_df = get_prediction(
        true_path=args.true_path,
        prediction_path=args.prediction_path,
        in_mark_down=args.markdown,
        metrics=metrics,
    )

    value_cols = EVAL_COLS_UNIQUE + ["rater_id"]
    averaged = (
        pred_df.groupby(value_cols, as_index=False, dropna=False)["value"]
        .mean()
        .loc[:, pred_df.columns]
    )
    averaged.to_csv(args.output_csv, index=False)
    print(f"Saved averaged predictions to {args.output_csv}")

    if args.score_json:
        score_dir = os.path.dirname(args.score_json)
        if score_dir:
            os.makedirs(score_dir, exist_ok=True)
        scores = score_predictions(true_df=true_df, pred_df=averaged, metrics=metrics)
        with open(args.score_json, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        print(f"Saved scores to {args.score_json}")


if __name__ == "__main__":
    main()
