#!/usr/bin/env python
"""Evaluate fixed-prefix bootstrap averages for with-gold/no-gold raw outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import pandas as pd

from eval_en import EVAL_COLS_UNIQUE, score_correlations
from utils.eval.eval_json import get_prediction


SCORE_KEYS = [
    ("ALL", "ALL-en-ALL-mean"),
    ("disagree_flag", "ALL-en-disagree_flag-mean"),
    ("completeness", "ALL-en-completeness-mean"),
    ("factual-accuracy", "ALL-en-factual-accuracy-mean"),
    ("relevance", "ALL-en-relevance-mean"),
    ("writing-style", "ALL-en-writing-style-mean"),
    ("overall", "ALL-en-overall-mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with_raw", required=True)
    parser.add_argument("--without_raw", required=True)
    parser.add_argument("--template_csv", required=True)
    parser.add_argument("--gold_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", default="['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']")
    parser.add_argument("--max_x", type=int, default=7)
    parser.add_argument("--markdown", action="store_true", default=True)
    parser.add_argument("--skip_selected_x", action="store_true")
    return parser.parse_args()


def bootstrap_id(item: dict) -> int:
    return int(item["input"][0]["key"][0]["bootstrap_id"])


def write_score_json(pred_csv: Path, gold_df: pd.DataFrame, score_json: Path) -> dict:
    pred_df = pd.read_csv(pred_csv).drop_duplicates(subset=EVAL_COLS_UNIQUE)
    scores = score_correlations(gold_df, pred_df)
    score_json.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def aggregate_prefix(
    raw_items: list[dict],
    context: str,
    x: int,
    template_csv: str,
    gold_df: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
    markdown: bool,
) -> dict:
    context_dir = output_dir / context / f"x{x}"
    context_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix_path = context_dir / "raw_prefix.json"
    pred_csv = context_dir / "prediction.csv"
    score_json = context_dir / "prediction.json"

    prefix_items = [item for item in raw_items if bootstrap_id(item) < x]
    if not prefix_items:
        raise SystemExit(f"No raw items found for {context} x={x}")
    raw_prefix_path.write_text(json.dumps(prefix_items, ensure_ascii=False), encoding="utf-8")

    _, pred_df = get_prediction(
        prediction_path=str(raw_prefix_path),
        in_mark_down=markdown,
        metrics=metrics,
        true_path=template_csv,
    )
    value_cols = EVAL_COLS_UNIQUE + ["rater_id"]
    averaged = (
        pred_df.groupby(value_cols, as_index=False, dropna=False)["value"]
        .mean()
        .loc[:, pred_df.columns]
    )
    averaged.to_csv(pred_csv, index=False)
    scores = write_score_json(pred_csv, gold_df, score_json)

    row = {
        "context": context,
        "x": x,
        "n_raw_items": len(prefix_items),
        "prediction_csv": str(pred_csv),
        "score_json": str(score_json),
    }
    for col, key in SCORE_KEYS:
        row[col] = scores.get(key)
    return row


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def write_selected_x(rows: list[dict], output_dir: Path) -> None:
    by_x: dict[int, dict[str, float]] = {}
    for row in rows:
        by_x.setdefault(int(row["x"]), {})[str(row["context"])] = safe_float(row["ALL"])

    selection_rows = []
    for x in sorted(by_x):
        with_score = by_x[x].get("with_gold", float("nan"))
        without_score = by_x[x].get("without_gold", float("nan"))
        vals = [v for v in [with_score, without_score] if not math.isnan(v)]
        avg = sum(vals) / len(vals) if vals else float("nan")
        selection_rows.append(
            {
                "x": x,
                "with_gold_ALL": with_score,
                "without_gold_ALL": without_score,
                "average_ALL": avg,
            }
        )

    best = max(selection_rows, key=lambda row: row["average_ALL"])
    with (output_dir / "selected_x.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_x": int(best["x"]),
                "selection_rule": "maximize average of dev with_gold and without_gold ALL-en-ALL-mean using fixed prefix bootstrap_id 0..x-1",
                "rows": selection_rows,
            },
            f,
            indent=2,
        )

    with (output_dir / "x_selection_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "with_gold_ALL", "without_gold_ALL", "average_ALL"])
        writer.writeheader()
        writer.writerows(selection_rows)


def main() -> None:
    args = parse_args()
    metrics = ast.literal_eval(args.metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_df = pd.read_csv(args.gold_csv)
    rows = []

    for context, raw_path in [
        ("with_gold", Path(args.with_raw)),
        ("without_gold", Path(args.without_raw)),
    ]:
        raw_items = json.loads(raw_path.read_text(encoding="utf-8"))
        for x in range(1, args.max_x + 1):
            rows.append(
                aggregate_prefix(
                    raw_items=raw_items,
                    context=context,
                    x=x,
                    template_csv=args.template_csv,
                    gold_df=gold_df,
                    metrics=metrics,
                    output_dir=output_dir,
                    markdown=args.markdown,
                )
            )

    sweep_csv = output_dir / "bootstrap_x_sweep.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["context", "x", "n_raw_items", "prediction_csv", "score_json"] + [col for col, _ in SCORE_KEYS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not args.skip_selected_x:
        write_selected_x(rows, output_dir)
    print(f"Saved bootstrap prefix sweep to {sweep_csv}")
    if not args.skip_selected_x:
        print(f"Saved selected x to {output_dir / 'selected_x.json'}")


if __name__ == "__main__":
    main()
