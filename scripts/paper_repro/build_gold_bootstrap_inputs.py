#!/usr/bin/env python3
"""Build paper-style Qwen with-gold bootstrap prompts without editing the source repo."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_METRICS = [
    "disagree_flag",
    "completeness",
    "factual-accuracy",
    "relevance",
    "writing-style",
    "overall",
]


def load_original_functions(repo_dir: Path, data_cwd: Path):
    os.chdir(data_cwd)
    sys.path.insert(0, str(repo_dir))
    make_shot = importlib.import_module("exp.few_shot.scripts.make_shot")
    get_data = importlib.import_module("utils.dataset_helper.get_data")
    return make_shot.prepare_gold_texts_shot, make_shot.make_identity, get_data.grouping


def sample_grouped_rows(df: pd.DataFrame, grouping_fn, shot_num: int, rng: random.Random) -> pd.DataFrame:
    grouped = grouping_fn(df, group_type=["metric"])
    keys = list(grouped.groups.keys())
    if not keys:
        raise ValueError("Shot pool is empty after filtering.")
    chosen = rng.choices(keys, k=shot_num)
    parts = []
    for copy_id, key in enumerate(chosen):
        part = grouped.get_group(key).copy()
        part["bootstrap_copy_id"] = copy_id
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def read_and_filter(path: Path, lang: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if lang and "lang" in df.columns:
        df = df[df["lang"] == lang].copy()
    return df.reset_index(drop=True)


def build_fold_input(
    *,
    repo_dir: Path,
    folded_csv: Path,
    output_path: Path,
    folds: list[int],
    shot_num: int,
    bootstrap_num: int,
    seed: int,
    sample_n_per_fold: int | None,
    data_cwd: Path,
    metrics: list[str],
    lang: str | None,
) -> None:
    prepare_gold_texts_shot, make_identity, grouping = load_original_functions(repo_dir, data_cwd)
    df = read_and_filter(folded_csv, lang=lang)
    if "fold" not in df.columns:
        raise ValueError(f"{folded_csv} must contain a fold column for dev OOF mode.")

    llm_input = []
    for fold in folds:
        shot_pool = df[df["fold"] != fold].copy()
        infer_df = df[df["fold"] == fold].copy()
        if sample_n_per_fold:
            infer_groups = list(grouping(infer_df, group_type=["metric"]))
            rng_sample = random.Random(seed + fold * 100003)
            rng_sample.shuffle(infer_groups)
            infer_df = pd.concat([g for _, g in infer_groups[:sample_n_per_fold]], ignore_index=True)
        infer_grouped = grouping(infer_df, group_type=["metric"])
        rng = random.Random(seed + fold * 1009)
        for bootstrap_id in tqdm(range(bootstrap_num), desc=f"fold {fold}"):
            shot_sample = sample_grouped_rows(shot_pool, grouping, shot_num, rng)
            shot_sample["bootstrap_id"] = bootstrap_id
            for _, infer in infer_grouped:
                prompt = prepare_gold_texts_shot(infer_df=infer, shot_df=shot_sample, metrics=metrics)
                key = make_identity(infer)
                key[0]["fold"] = fold
                key[0]["bootstrap_id"] = bootstrap_id
                prompt["key"] = key
                llm_input.append(prompt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(llm_input)} prompt items to {output_path}")


def build_test_input(
    *,
    repo_dir: Path,
    shot_csv: Path,
    infer_csv: Path,
    output_path: Path,
    shot_num: int,
    bootstrap_num: int,
    seed: int,
    sample_n: int | None,
    data_cwd: Path,
    metrics: list[str],
    lang: str | None,
) -> None:
    prepare_gold_texts_shot, make_identity, grouping = load_original_functions(repo_dir, data_cwd)
    shot_pool = read_and_filter(shot_csv, lang=lang)
    infer_df = read_and_filter(infer_csv, lang=lang)
    if sample_n:
        infer_groups = list(grouping(infer_df, group_type=["metric"]))
        rng_sample = random.Random(seed)
        rng_sample.shuffle(infer_groups)
        infer_df = pd.concat([g for _, g in infer_groups[:sample_n]], ignore_index=True)

    infer_grouped = grouping(infer_df, group_type=["metric"])
    rng = random.Random(seed)
    llm_input = []
    for bootstrap_id in tqdm(range(bootstrap_num), desc="test"):
        shot_sample = sample_grouped_rows(shot_pool, grouping, shot_num, rng)
        shot_sample["bootstrap_id"] = bootstrap_id
        for _, infer in infer_grouped:
            prompt = prepare_gold_texts_shot(infer_df=infer, shot_df=shot_sample, metrics=metrics)
            key = make_identity(infer)
            key[0]["bootstrap_id"] = bootstrap_id
            prompt["key"] = key
            llm_input.append(prompt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(llm_input)} prompt items to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["dev-oof", "test"])
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--folded-csv", type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--shot-csv", type=Path)
    parser.add_argument("--infer-csv", type=Path)
    parser.add_argument("--shot-num", type=int, default=20)
    parser.add_argument("--bootstrap-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--data-cwd", type=Path, default=Path.cwd())
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--sample-n-per-fold", type=int)
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--lang", default="en", help="Language code to keep; use 'all' to disable filtering.")
    args = parser.parse_args()
    metrics = ast.literal_eval(args.metrics)
    lang = None if args.lang == "all" else args.lang

    if args.mode == "dev-oof":
        if not args.folded_csv:
            raise SystemExit("--folded-csv is required for dev-oof")
        build_fold_input(
            repo_dir=args.repo_dir,
            folded_csv=args.folded_csv,
            output_path=args.output_path,
            folds=[int(x) for x in args.folds.split(",") if x.strip()],
            shot_num=args.shot_num,
            bootstrap_num=args.bootstrap_num,
            seed=args.seed,
            sample_n_per_fold=args.sample_n_per_fold,
            data_cwd=args.data_cwd,
            metrics=metrics,
            lang=lang,
        )
    else:
        if not args.shot_csv or not args.infer_csv:
            raise SystemExit("--shot-csv and --infer-csv are required for test")
        build_test_input(
            repo_dir=args.repo_dir,
            shot_csv=args.shot_csv,
            infer_csv=args.infer_csv,
            output_path=args.output_path,
            shot_num=args.shot_num,
            bootstrap_num=args.bootstrap_num,
            seed=args.seed,
            sample_n=args.sample_n,
            data_cwd=args.data_cwd,
            metrics=metrics,
            lang=lang,
        )


if __name__ == "__main__":
    main()
