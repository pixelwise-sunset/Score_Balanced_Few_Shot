#!/usr/bin/env python3
"""Build dev OOF score-balanced softmax few-shot inputs.

For each validation fold, examples from the other folds form the shot pool and
the held-out fold is used for inference. The output is one aggregate input JSON
that can be run once through Qwen and later aggregated against the official dev
gold CSV.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


KEY_COLS = ["dataset", "encounter_id", "lang", "candidate", "candidate_author_id"]
DEFAULT_METRICS = [
    "disagree_flag",
    "completeness",
    "factual-accuracy",
    "relevance",
    "writing-style",
    "overall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--data-cwd", default=Path.cwd(), type=Path)
    parser.add_argument("--folded-csv", default="datasets/aligned_en_folded.csv", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--shot-num", type=int, default=20)
    parser.add_argument("--bootstrap-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--en-only", action="store_true", help="Deprecated alias for --lang en.")
    parser.add_argument("--lang", default="en", help="Language code to keep; use 'all' to disable filtering.")
    return parser.parse_args()


def load_original_functions(repo_dir: Path, data_cwd: Path):
    repo_dir = repo_dir.resolve()
    data_cwd = data_cwd.resolve()
    os.chdir(data_cwd)
    sys.path.insert(0, str(repo_dir))
    make_shot = importlib.import_module("exp.few_shot.scripts.make_shot")
    get_data = importlib.import_module("utils.dataset_helper.get_data")
    return (
        make_shot.prepare_gold_texts_shot,
        make_shot.make_identity,
        get_data.grouping,
        get_data.grouping_with_gold,
    )


def metric_bin(value: float) -> str:
    if value <= 0.0:
        return "low"
    if value >= 1.0:
        return "high"
    return "mid"


def group_signature(group: pd.DataFrame) -> tuple:
    row = group.iloc[0]
    return tuple(row[col] for col in KEY_COLS)


def labels_by_metric(group: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    labels = {}
    for metric in metrics:
        values = group.loc[group["metric"] == metric, "label"]
        if values.empty:
            raise ValueError(f"Missing metric {metric} in shot group {group_signature(group)}")
        labels[metric] = float(values.iloc[0])
    return labels


def score_balance_candidate(
    group: pd.DataFrame,
    selected: list[pd.DataFrame],
    coverage: dict[tuple[str, str], int],
    metrics: list[str],
) -> float:
    score = 0.0
    labels = labels_by_metric(group, metrics)
    for metric, value in labels.items():
        score += 1.0 / (1.0 + coverage[(metric, metric_bin(value))])
    dataset = str(group["dataset"].iloc[0])
    if selected and dataset not in {str(item["dataset"].iloc[0]) for item in selected}:
        score += 0.25
    return score


def softmax_sample(items: list[tuple[int, float]], temperature: float, rng: random.Random) -> tuple[int, dict[int, float]]:
    if temperature <= 0:
        raise ValueError(f"--softmax-temperature must be positive, got {temperature}")
    max_score = max(score for _, score in items)
    weights = [(idx, math.exp((score - max_score) / temperature)) for idx, score in items]
    total = sum(weight for _, weight in weights)
    probs = {idx: weight / total for idx, weight in weights} if total > 0 else {
        idx: 1.0 / len(weights) for idx, _ in weights
    }
    threshold = rng.random()
    cumulative = 0.0
    for idx, _ in weights:
        cumulative += probs[idx]
        if threshold <= cumulative:
            return idx, probs
    return weights[-1][0], probs


def select_score_balanced_softmax(
    groups: list[pd.DataFrame],
    metrics: list[str],
    shot_num: int,
    rng: random.Random,
    temperature: float,
) -> tuple[list[pd.DataFrame], list[dict]]:
    shuffled = list(groups)
    rng.shuffle(shuffled)
    selected: list[pd.DataFrame] = []
    selected_sigs: set[tuple] = set()
    trace: list[dict] = []
    coverage = {(metric, bucket): 0 for metric in metrics for bucket in ["low", "mid", "high"]}

    while len(selected) < min(shot_num, len(shuffled)):
        scored = []
        for idx, group in enumerate(shuffled):
            sig = group_signature(group)
            if sig not in selected_sigs:
                scored.append((idx, score_balance_candidate(group, selected, coverage, metrics)))
        if not scored:
            break

        chosen_idx, probs = softmax_sample(scored, temperature, rng)
        chosen = shuffled[chosen_idx]
        selected.append(chosen)
        selected_sigs.add(group_signature(chosen))
        labels = labels_by_metric(chosen, metrics)
        for metric, value in labels.items():
            coverage[(metric, metric_bin(value))] += 1
        trace.append(
            {
                "signature": group_signature(chosen),
                "balance_score": dict(scored)[chosen_idx],
                "sampling_probability": probs[chosen_idx],
                "candidate_pool_size": len(scored),
            }
        )
    return selected, trace


def main() -> None:
    args = parse_args()
    metrics = ast.literal_eval(args.metrics)
    folds = [int(item) for item in args.folds.split(",") if item.strip()]
    prepare_gold_texts_shot, make_identity, grouping, grouping_with_gold = load_original_functions(
        args.repo_dir,
        args.data_cwd,
    )

    df = pd.read_csv(args.folded_csv)
    lang = "en" if args.en_only else args.lang
    if lang != "all":
        df = df[df["lang"] == lang].copy()
    if "fold" not in df.columns:
        raise SystemExit(f"{args.folded_csv} must contain a fold column")

    llm_input = []
    fold_counts = {}
    for fold in folds:
        shot_df = df[df["fold"].astype(int) != fold].copy()
        infer_df = df[df["fold"].astype(int) == fold].copy()
        shot_groups = [group.copy() for _, group in grouping_with_gold(shot_df, group_type=["metric"])]
        infer_grouped = grouping(infer_df, group_type=["metric"])
        fold_counts[str(fold)] = {"shot_groups": len(shot_groups), "infer_groups": len(infer_grouped)}

        for bootstrap_id in tqdm(range(args.bootstrap_num), desc=f"fold {fold}"):
            rng = random.Random(args.seed + fold * 100003 + bootstrap_id * 9176)
            selected_groups, selection_trace = select_score_balanced_softmax(
                groups=shot_groups,
                metrics=metrics,
                shot_num=args.shot_num,
                rng=rng,
                temperature=args.softmax_temperature,
            )
            selected_df = pd.concat(selected_groups, ignore_index=True)
            selected_df["bootstrap_id"] = bootstrap_id
            trace_by_sig = {tuple(item["signature"]): item for item in selection_trace}
            metadata = []
            for rank, group in enumerate(selected_groups, start=1):
                labels = labels_by_metric(group, metrics)
                trace = trace_by_sig.get(group_signature(group), {})
                metadata.append(
                    {
                        "dataset": str(group["dataset"].iloc[0]),
                        "encounter_id": str(group["encounter_id"].iloc[0]),
                        "candidate_author_id": str(group["candidate_author_id"].iloc[0]),
                        "rank": rank,
                        "fold": fold,
                        "bootstrap_id": bootstrap_id,
                        "label_bins": {metric: metric_bin(value) for metric, value in labels.items()},
                        "labels": labels,
                        "balance_score": trace.get("balance_score"),
                        "sampling_probability": trace.get("sampling_probability"),
                        "candidate_pool_size": trace.get("candidate_pool_size"),
                    }
                )

            for _, infer in infer_grouped:
                prompt = prepare_gold_texts_shot(infer_df=infer, shot_df=selected_df, metrics=metrics)
                key = make_identity(infer)
                key[0]["fold"] = fold
                key[0]["bootstrap_id"] = bootstrap_id
                key[0]["selection_strategy"] = "score_balanced_softmax"
                prompt["key"] = key
                prompt["shot_metadata"] = metadata
                llm_input.append(prompt)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(llm_input, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "method": "score_balanced_softmax20x7_dev_oof",
        "folded_csv": str(args.folded_csv),
        "folds": folds,
        "shot_num": args.shot_num,
        "bootstrap_num": args.bootstrap_num,
        "seed": args.seed,
        "softmax_temperature": args.softmax_temperature,
        "metrics": metrics,
        "lang": lang,
        "fold_counts": fold_counts,
        "prompt_items": len(llm_input),
    }
    args.output_json.with_name("input_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(llm_input)} prompt items to {args.output_json}")


if __name__ == "__main__":
    main()
