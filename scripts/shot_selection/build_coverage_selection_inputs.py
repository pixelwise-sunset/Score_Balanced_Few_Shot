#!/usr/bin/env python3
"""Build coverage-selection few-shot inputs for English ablations.

This script supports two strategies:

- metric_bin_coverage: one global 20-shot set per bootstrap/fold.
- per_sample_coverage_similarity: target-specific 20-shot sets combining
  coverage pressure and semantic similarity.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
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
BUCKETS = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--data-cwd", default=Path.cwd(), type=Path)
    parser.add_argument("--mode", choices=["dev_oof", "test"], required=True)
    parser.add_argument("--strategy", choices=["metric_bin_coverage", "per_sample_coverage_similarity"], required=True)
    parser.add_argument("--folded-csv", default="datasets/aligned_en_folded.csv", type=Path)
    parser.add_argument("--shot-csv", type=Path)
    parser.add_argument("--infer-csv", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--shot-num", type=int, default=20)
    parser.add_argument("--bootstrap-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--en-only", action="store_true", help="Deprecated alias for --lang en.")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--embedding-model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    parser.add_argument("--embedding-backend", choices=["hf_mean_pool", "hf_cls_pool"], default="hf_mean_pool")
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--similarity-weight", type=float, default=0.5)
    parser.add_argument("--dataset-diversity-bonus", type=float, default=0.25)
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


def make_encoder(args: argparse.Namespace):
    from utils.embedding.hf_mean_pool import HFCLSPoolTextEncoder, HFMeanPoolTextEncoder

    cls = HFCLSPoolTextEncoder if args.embedding_backend == "hf_cls_pool" else HFMeanPoolTextEncoder
    return cls(args.embedding_model, max_length=args.embedding_max_length)


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


def bins_by_metric(group: pd.DataFrame, metrics: list[str]) -> dict[str, str]:
    labels = labels_by_metric(group, metrics)
    return {metric: metric_bin(value) for metric, value in labels.items()}


def observed_target_bins(groups: list[pd.DataFrame], metrics: list[str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for group in groups:
        for metric, bucket in bins_by_metric(group, metrics).items():
            out.add((metric, bucket))
    return out


def group_bins(group: pd.DataFrame, metrics: list[str]) -> set[tuple[str, str]]:
    return set(bins_by_metric(group, metrics).items())


def coverage_counts_template(metrics: list[str]) -> dict[tuple[str, str], int]:
    return {(metric, bucket): 0 for metric in metrics for bucket in BUCKETS}


def update_coverage(coverage: dict[tuple[str, str], int], group: pd.DataFrame, metrics: list[str]) -> None:
    for item in group_bins(group, metrics):
        coverage[item] += 1


def rarity_score(group: pd.DataFrame, coverage: dict[tuple[str, str], int], metrics: list[str]) -> float:
    return sum(1.0 / (1.0 + coverage[item]) for item in group_bins(group, metrics))


def dataset_bonus(group: pd.DataFrame, selected: list[pd.DataFrame], bonus: float) -> float:
    if not selected:
        return 0.0
    dataset = str(group["dataset"].iloc[0])
    selected_datasets = {str(item["dataset"].iloc[0]) for item in selected}
    return bonus if dataset not in selected_datasets else 0.0


def sorted_target_bins(target_bins: set[tuple[str, str]], metrics: list[str]) -> list[dict[str, str]]:
    metric_order = {metric: idx for idx, metric in enumerate(metrics)}
    bucket_order = {bucket: idx for idx, bucket in enumerate(BUCKETS)}
    return [
        {"metric": metric, "bin": bucket}
        for metric, bucket in sorted(
            target_bins,
            key=lambda item: (metric_order.get(item[0], 999), bucket_order.get(item[1], 999)),
        )
    ]


def coverage_as_json(coverage: dict[tuple[str, str], int], metrics: list[str]) -> dict[str, dict[str, int]]:
    return {metric: {bucket: int(coverage[(metric, bucket)]) for bucket in BUCKETS} for metric in metrics}


def select_metric_bin_coverage(
    groups: list[pd.DataFrame],
    metrics: list[str],
    shot_num: int,
    rng: random.Random,
    diversity_bonus: float,
) -> tuple[list[pd.DataFrame], list[dict], dict]:
    selected: list[pd.DataFrame] = []
    selected_sigs: set[tuple] = set()
    trace: list[dict] = []
    coverage = coverage_counts_template(metrics)
    target_bins = observed_target_bins(groups, metrics)

    while len(selected) < min(shot_num, len(groups)):
        covered = {item for item, count in coverage.items() if count > 0}
        uncovered = target_bins - covered
        phase = "cover_uncovered" if uncovered else "fill_balanced"
        best_idx = None
        best_key = None
        best_meta = None
        for idx, group in enumerate(groups):
            sig = group_signature(group)
            if sig in selected_sigs:
                continue
            bins = group_bins(group, metrics)
            uncovered_gain = len(bins & uncovered) if uncovered else 0
            rarity = rarity_score(group, coverage, metrics)
            bonus = dataset_bonus(group, selected, diversity_bonus)
            rand = rng.random() * 1e-9
            if phase == "cover_uncovered":
                sort_key = (uncovered_gain, rarity, bonus, rand)
            else:
                sort_key = (rarity, bonus, rand)
            if best_key is None or sort_key > best_key:
                best_idx = idx
                best_key = sort_key
                best_meta = {
                    "selection_phase": phase,
                    "uncovered_gain": uncovered_gain,
                    "coverage_component": rarity / max(1, len(metrics)),
                    "rarity_score": rarity,
                    "dataset_diversity_bonus": bonus,
                }
        if best_idx is None:
            break

        chosen = groups[best_idx]
        selected.append(chosen)
        selected_sigs.add(group_signature(chosen))
        labels = labels_by_metric(chosen, metrics)
        update_coverage(coverage, chosen, metrics)
        trace.append(
            {
                "signature": group_signature(chosen),
                "rank": len(selected),
                "labels": labels,
                "label_bins": {metric: metric_bin(value) for metric, value in labels.items()},
                **(best_meta or {}),
            }
        )

    covered_final = {item for item, count in coverage.items() if count > 0}
    summary = {
        "target_bins": sorted_target_bins(target_bins, metrics),
        "missing_target_bins": sorted_target_bins(target_bins - covered_final, metrics),
        "coverage_counts": coverage_as_json(coverage, metrics),
    }
    return selected, trace, summary


def normalize_similarity(similarities: np.ndarray) -> np.ndarray:
    if similarities.size == 0:
        return similarities.astype(float)
    min_val = float(np.min(similarities))
    max_val = float(np.max(similarities))
    if max_val <= min_val:
        return np.zeros_like(similarities, dtype=float)
    return (similarities.astype(float) - min_val) / (max_val - min_val)


def select_per_sample_coverage_similarity(
    groups: list[pd.DataFrame],
    similarities: np.ndarray,
    metrics: list[str],
    shot_num: int,
    rng: random.Random,
    coverage_weight: float,
    similarity_weight: float,
    diversity_bonus: float,
) -> tuple[list[pd.DataFrame], list[dict], dict]:
    selected: list[pd.DataFrame] = []
    selected_sigs: set[tuple] = set()
    trace: list[dict] = []
    coverage = coverage_counts_template(metrics)
    target_bins = observed_target_bins(groups, metrics)
    sim_norm = normalize_similarity(similarities)

    while len(selected) < min(shot_num, len(groups)):
        best_idx = None
        best_key = None
        best_meta = None
        for idx, group in enumerate(groups):
            sig = group_signature(group)
            if sig in selected_sigs:
                continue
            rarity = rarity_score(group, coverage, metrics)
            coverage_component = rarity / max(1, len(metrics))
            similarity_component = float(sim_norm[idx])
            hybrid_score = coverage_weight * coverage_component + similarity_weight * similarity_component
            bonus = dataset_bonus(group, selected, diversity_bonus)
            rand = rng.random() * 1e-9
            sort_key = (hybrid_score, bonus, rand)
            if best_key is None or sort_key > best_key:
                best_idx = idx
                best_key = sort_key
                best_meta = {
                    "selection_phase": "hybrid_weighted",
                    "coverage_component": coverage_component,
                    "similarity": float(similarities[idx]),
                    "similarity_component": similarity_component,
                    "hybrid_score": hybrid_score,
                    "dataset_diversity_bonus": bonus,
                }
        if best_idx is None:
            break

        chosen = groups[best_idx]
        selected.append(chosen)
        selected_sigs.add(group_signature(chosen))
        labels = labels_by_metric(chosen, metrics)
        update_coverage(coverage, chosen, metrics)
        trace.append(
            {
                "signature": group_signature(chosen),
                "rank": len(selected),
                "labels": labels,
                "label_bins": {metric: metric_bin(value) for metric, value in labels.items()},
                **(best_meta or {}),
            }
        )

    covered_final = {item for item, count in coverage.items() if count > 0}
    summary = {
        "target_bins": sorted_target_bins(target_bins, metrics),
        "missing_target_bins": sorted_target_bins(target_bins - covered_final, metrics),
        "coverage_counts": coverage_as_json(coverage, metrics),
    }
    return selected, trace, summary


def semantic_text(group: pd.DataFrame) -> str:
    row = group.iloc[0]
    parts = [
        f"Query: {row.get('query_text', '')}",
        f"LLM response: {row.get('candidate', '')}",
    ]
    caption = row.get("image_caption", "")
    if isinstance(caption, str) and caption.strip():
        parts.append(f"Image caption: {caption}")
    return "\n".join(parts)


def build_prompt(
    infer_group: pd.DataFrame,
    selected_groups: list[pd.DataFrame],
    trace: list[dict],
    summary: dict,
    metrics: list[str],
    prepare_gold_texts_shot,
) -> dict:
    selected_df = pd.concat(selected_groups, ignore_index=True) if selected_groups else infer_group.iloc[0:0].copy()
    prompt = prepare_gold_texts_shot(infer_df=infer_group, shot_df=selected_df, metrics=metrics)
    prompt["shot_metadata"] = trace
    prompt["coverage_summary"] = summary
    return prompt


def filter_lang(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    if lang == "all":
        return df.copy()
    return df[df["lang"] == lang].copy()


def build_for_split(
    *,
    shot_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    fold: int | None,
    args: argparse.Namespace,
    metrics: list[str],
    grouping,
    grouping_with_gold,
    prepare_gold_texts_shot,
    make_identity,
) -> tuple[list[dict], dict]:
    shot_groups = [group.copy() for _, group in grouping_with_gold(shot_df, group_type=["metric"])]
    infer_groups = [group.copy() for _, group in grouping(infer_df, group_type=["metric"])]
    if args.sample_n:
        infer_groups = infer_groups[: args.sample_n]

    shot_emb = None
    infer_emb = None
    if args.strategy == "per_sample_coverage_similarity":
        encoder = make_encoder(args)
        shot_emb = np.asarray(
            encoder.encode(
                [semantic_text(group) for group in shot_groups],
                batch_size=args.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        infer_emb = np.asarray(
            encoder.encode(
                [semantic_text(group) for group in infer_groups],
                batch_size=args.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    items: list[dict] = []
    global_by_bootstrap: dict[int, tuple[list[pd.DataFrame], list[dict], dict]] = {}
    iterator = tqdm(infer_groups, desc=f"fold {fold}" if fold is not None else "test")
    for infer_idx, infer_group in enumerate(iterator):
        for bootstrap_id in range(args.bootstrap_num):
            rng = random.Random(args.seed + (fold or 0) * 100003 + infer_idx * 1009 + bootstrap_id * 9176)
            if args.strategy == "metric_bin_coverage":
                if bootstrap_id not in global_by_bootstrap:
                    global_rng = random.Random(args.seed + (fold or 0) * 100003 + bootstrap_id * 9176)
                    global_by_bootstrap[bootstrap_id] = select_metric_bin_coverage(
                        groups=shot_groups,
                        metrics=metrics,
                        shot_num=args.shot_num,
                        rng=global_rng,
                        diversity_bonus=args.dataset_diversity_bonus,
                    )
                selected_groups, trace, summary = global_by_bootstrap[bootstrap_id]
            else:
                assert shot_emb is not None and infer_emb is not None
                similarities = shot_emb @ infer_emb[infer_idx]
                selected_groups, trace, summary = select_per_sample_coverage_similarity(
                    groups=shot_groups,
                    similarities=similarities,
                    metrics=metrics,
                    shot_num=args.shot_num,
                    rng=rng,
                    coverage_weight=args.coverage_weight,
                    similarity_weight=args.similarity_weight,
                    diversity_bonus=args.dataset_diversity_bonus,
                )

            prompt = build_prompt(
                infer_group=infer_group,
                selected_groups=selected_groups,
                trace=trace,
                summary=summary,
                metrics=metrics,
                prepare_gold_texts_shot=prepare_gold_texts_shot,
            )
            key = make_identity(infer_group)
            if fold is not None:
                key[0]["fold"] = fold
            key[0]["bootstrap_id"] = bootstrap_id
            key[0]["selection_strategy"] = args.strategy
            prompt["key"] = key
            items.append(prompt)

    split_counts = {"shot_groups": len(shot_groups), "infer_groups": len(infer_groups)}
    return items, split_counts


def main() -> None:
    args = parse_args()
    metrics = ast.literal_eval(args.metrics)
    lang = "en" if args.en_only else args.lang
    prepare_gold_texts_shot, make_identity, grouping, grouping_with_gold = load_original_functions(
        args.repo_dir,
        args.data_cwd,
    )

    all_items: list[dict] = []
    split_counts = {}
    if args.mode == "dev_oof":
        df = filter_lang(pd.read_csv(args.folded_csv), lang)
        folds = [int(item) for item in args.folds.split(",") if item.strip()]
        for fold in folds:
            shot_df = df[df["fold"].astype(int) != fold].copy()
            infer_df = df[df["fold"].astype(int) == fold].copy()
            items, counts = build_for_split(
                shot_df=shot_df,
                infer_df=infer_df,
                fold=fold,
                args=args,
                metrics=metrics,
                grouping=grouping,
                grouping_with_gold=grouping_with_gold,
                prepare_gold_texts_shot=prepare_gold_texts_shot,
                make_identity=make_identity,
            )
            all_items.extend(items)
            split_counts[str(fold)] = counts
    else:
        if not args.shot_csv or not args.infer_csv:
            raise SystemExit("--shot-csv and --infer-csv are required for --mode test")
        shot_df = filter_lang(pd.read_csv(args.shot_csv), lang)
        infer_df = filter_lang(pd.read_csv(args.infer_csv), lang)
        items, counts = build_for_split(
            shot_df=shot_df,
            infer_df=infer_df,
            fold=None,
            args=args,
            metrics=metrics,
            grouping=grouping,
            grouping_with_gold=grouping_with_gold,
            prepare_gold_texts_shot=prepare_gold_texts_shot,
            make_identity=make_identity,
        )
        all_items.extend(items)
        split_counts["test"] = counts

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_items, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "method": args.strategy,
        "mode": args.mode,
        "output_json": str(args.output_json),
        "metrics": metrics,
        "lang": lang,
        "shot_num": args.shot_num,
        "bootstrap_num": args.bootstrap_num,
        "seed": args.seed,
        "coverage_weight": args.coverage_weight,
        "similarity_weight": args.similarity_weight,
        "embedding_model": args.embedding_model if args.strategy == "per_sample_coverage_similarity" else None,
        "embedding_backend": args.embedding_backend if args.strategy == "per_sample_coverage_similarity" else None,
        "semantic_text": "query_text + candidate + image_caption",
        "split_counts": split_counts,
        "prompt_items": len(all_items),
    }
    args.output_json.with_name("input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(all_items)} prompt items to {args.output_json}")


if __name__ == "__main__":
    main()
