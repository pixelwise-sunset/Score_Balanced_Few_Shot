#!/usr/bin/env python3
"""Parse, average, and score paper-style bootstrap predictions."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "disagree_flag",
    "completeness",
    "factual-accuracy",
    "relevance",
    "writing-style",
    "overall",
]
EVAL_COLS_UNIQUE = [
    "dataset",
    "encounter_id",
    "lang",
    "candidate",
    "candidate_author_id",
    "metric",
]


def parse_response(text: str) -> dict[str, float]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    parsed = json.loads(match.group(1))
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Empty JSON list in response.")
        parsed = parsed[0]
    out = {}
    for key, value in parsed.items():
        norm_key = key.replace("_", "-")
        if norm_key in {"disagree-flag", "disagree"}:
            norm_key = "disagree_flag"
        elif norm_key == "factual-accuracy" or key == "factual_accuracy":
            norm_key = "factual-accuracy"
        elif norm_key == "writing-style" or key == "writing_style":
            norm_key = "writing-style"
        out[norm_key] = float(value)
    return out


def load_predictions(raw_json: Path, template_csv: Path, metrics: list[str], lang: str | None) -> pd.DataFrame:
    with raw_json.open(encoding="utf-8") as f:
        outputs = json.load(f)
    template = pd.read_csv(template_csv)
    if "official_idx" in template.columns:
        template = template.drop(columns=["official_idx"])
    if lang:
        template = template[template["lang"] == lang].copy()

    rows = []
    failures = []
    for idx, item in enumerate(outputs):
        try:
            response = item["response"][0]
            scores = parse_response(response)
            key = item["input"][0]["key"][0]
        except Exception as exc:
            failures.append({"index": idx, "error": str(exc), "response": item.get("response")})
            continue

        for metric in metrics:
            if metric not in scores:
                failures.append({"index": idx, "error": f"missing {metric}", "response": response})
                continue
            mask = (
                (template["dataset"] == key["dataset"])
                & (template["encounter_id"] == key["encounter_id"])
                & (template["lang"] == key["lang"])
                & (template["candidate_author_id"] == key["candidate_author_id"])
                & (template["metric"] == metric)
            )
            slice_df = template.loc[mask].copy()
            if slice_df.empty:
                failures.append({"index": idx, "error": f"no template row for {key} metric={metric}"})
                continue
            slice_df["value"] = scores[metric]
            slice_df["bootstrap_id"] = key.get("bootstrap_id", 0)
            if "fold" in key:
                slice_df["fold"] = key["fold"]
            rows.append(slice_df)

    if not rows:
        raise RuntimeError(f"No predictions parsed from {raw_json}")
    pred = pd.concat(rows, ignore_index=True)
    if failures:
        failure_path = raw_json.with_name(raw_json.stem + "_parse_failures.json")
        with failure_path.open("w", encoding="utf-8") as f:
            json.dump(failures[:500], f, indent=2, ensure_ascii=False)
        print(f"[WARN] {len(failures)} parse/template failures; wrote sample to {failure_path}")
    return pred


def average_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in pred.columns if c not in {"value", "bootstrap_id", "fold"}]
    averaged = pred.groupby(keep_cols, as_index=False, dropna=False)["value"].mean()
    return averaged.loc[:, [c for c in pred.columns if c in averaged.columns]]


def score_with_eval(eval_py: Path, gold_csv: Path, pred_csv: Path, score_json: Path) -> None:
    import subprocess

    score_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(eval_py), str(gold_csv), str(pred_csv), str(score_json)],
        check=True,
    )


def compact_scores(score_json: Path) -> None:
    if not score_json.exists():
        return
    with score_json.open(encoding="utf-8") as f:
        scores = json.load(f)
    keys = [key for key in scores if key.startswith("ALL-") and key.endswith("-mean")]
    print("[SCORES]")
    for key in keys:
        val = scores.get(key)
        if isinstance(val, float) and math.isnan(val):
            val = "NaN"
        print(f"{key}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", required=True, type=Path)
    parser.add_argument("--template-csv", required=True, type=Path)
    parser.add_argument("--gold-csv", required=True, type=Path)
    parser.add_argument("--eval-py", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--all-runs-csv", type=Path)
    parser.add_argument("--score-json", required=True, type=Path)
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--lang", default="en", help="Language code to keep; use 'all' to disable filtering.")
    args = parser.parse_args()
    metrics = ast.literal_eval(args.metrics)
    lang = None if args.lang == "all" else args.lang

    pred = load_predictions(args.raw_json, args.template_csv, metrics=metrics, lang=lang)
    if args.all_runs_csv:
        args.all_runs_csv.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(args.all_runs_csv, index=False)
        print(f"Saved all bootstrap rows to {args.all_runs_csv} ({len(pred)} rows)")

    averaged = average_predictions(pred)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    averaged.to_csv(args.output_csv, index=False)
    print(f"Saved averaged prediction CSV to {args.output_csv} ({len(averaged)} rows)")
    score_with_eval(args.eval_py, args.gold_csv, args.output_csv, args.score_json)
    compact_scores(args.score_json)


if __name__ == "__main__":
    main()
