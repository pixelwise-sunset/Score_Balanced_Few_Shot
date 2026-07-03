import ast
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp.few_shot.scripts.make_shot import make_identity, prepare_gold_texts_shot
from utils.dataset_helper.get_data import grouping, grouping_with_gold


KEY_COLS = ["dataset", "encounter_id", "lang", "candidate", "candidate_author_id"]


def metric_bin(value: float) -> str:
    if value <= 0.0:
        return "low"
    if value >= 1.0:
        return "high"
    return "mid"


def group_examples(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [group.copy() for _, group in grouping_with_gold(df, group_type=["metric"])]


def group_signature(group: pd.DataFrame) -> tuple:
    row = group.iloc[0]
    return tuple(row[col] for col in KEY_COLS)


def labels_by_metric(group: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    out = {}
    for metric in metrics:
        values = group.loc[group["metric"] == metric, "label"]
        if values.empty:
            raise ValueError(f"Missing metric {metric} in group {group_signature(group)}")
        out[metric] = float(values.iloc[0])
    return out


def agreement_filter_keys(raw_path: str, metrics: list[str], threshold: float) -> set[tuple]:
    raw = pd.read_csv(raw_path)
    raw = raw[(raw["lang"] == "en") & (raw["metric"].isin(metrics))].copy()

    keep = set()
    for key, group in raw.groupby(KEY_COLS, dropna=False):
        disagreements = []
        complete = True
        for metric in metrics:
            metric_group = group[group["metric"] == metric]
            values = sorted(metric_group["value"].dropna().astype(float).tolist())
            if len(values) <= 1:
                continue
            if len(values) != 2:
                complete = False
                break
            disagreements.append(abs(values[0] - values[1]))
        if complete and (not disagreements or max(disagreements) <= threshold):
            keep.add(key)
    return keep


def select_score_balanced(
    groups: list[pd.DataFrame],
    metrics: list[str],
    shot_num: int,
    rng: random.Random,
) -> list[pd.DataFrame]:
    if shot_num <= 0:
        return []

    shuffled = list(groups)
    rng.shuffle(shuffled)
    selected: list[pd.DataFrame] = []
    selected_sigs: set[tuple] = set()
    coverage = {(metric, bucket): 0 for metric in metrics for bucket in ["low", "mid", "high"]}

    while len(selected) < min(shot_num, len(shuffled)):
        best_idx = None
        best_score = None

        for idx, group in enumerate(shuffled):
            sig = group_signature(group)
            if sig in selected_sigs:
                continue
            labels = labels_by_metric(group, metrics)
            score = 0.0
            for metric, value in labels.items():
                bucket = metric_bin(value)
                score += 1.0 / (1.0 + coverage[(metric, bucket)])
            dataset = str(group["dataset"].iloc[0])
            if selected and dataset not in {str(g["dataset"].iloc[0]) for g in selected}:
                score += 0.25
            tie_breaker = rng.random() * 1e-6
            score += tie_breaker

            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            break

        chosen = shuffled[best_idx]
        selected.append(chosen)
        selected_sigs.add(group_signature(chosen))
        for metric, value in labels_by_metric(chosen, metrics).items():
            coverage[(metric, metric_bin(value))] += 1

    return selected


def softmax_sample(items: list[tuple[int, float]], temperature: float, rng: random.Random) -> tuple[int, dict[int, float]]:
    if temperature <= 0:
        raise ValueError(f"softmax_temperature must be positive, got {temperature}")
    if not items:
        raise ValueError("Cannot sample from an empty item list")

    max_score = max(score for _, score in items)
    weights = [(idx, math.exp((score - max_score) / temperature)) for idx, score in items]
    total = sum(weight for _, weight in weights)
    if total <= 0:
        probs = {idx: 1.0 / len(weights) for idx, _ in weights}
    else:
        probs = {idx: weight / total for idx, weight in weights}

    threshold = rng.random()
    cumulative = 0.0
    for idx, _ in weights:
        cumulative += probs[idx]
        if threshold <= cumulative:
            return idx, probs
    return weights[-1][0], probs


def score_balance_candidate(
    group: pd.DataFrame,
    selected: list[pd.DataFrame],
    coverage: dict[tuple[str, str], int],
    metrics: list[str],
) -> float:
    labels = labels_by_metric(group, metrics)
    score = 0.0
    for metric, value in labels.items():
        bucket = metric_bin(value)
        score += 1.0 / (1.0 + coverage[(metric, bucket)])
    dataset = str(group["dataset"].iloc[0])
    if selected and dataset not in {str(g["dataset"].iloc[0]) for g in selected}:
        score += 0.25
    return score


def select_score_balanced_softmax(
    groups: list[pd.DataFrame],
    metrics: list[str],
    shot_num: int,
    rng: random.Random,
    temperature: float,
) -> tuple[list[pd.DataFrame], list[dict]]:
    if shot_num <= 0:
        return [], []

    shuffled = list(groups)
    rng.shuffle(shuffled)
    selected: list[pd.DataFrame] = []
    selected_sigs: set[tuple] = set()
    selection_trace: list[dict] = []
    coverage = {(metric, bucket): 0 for metric in metrics for bucket in ["low", "mid", "high"]}

    while len(selected) < min(shot_num, len(shuffled)):
        scored: list[tuple[int, float]] = []
        for idx, group in enumerate(shuffled):
            sig = group_signature(group)
            if sig in selected_sigs:
                continue
            scored.append(
                (
                    idx,
                    score_balance_candidate(
                        group=group,
                        selected=selected,
                        coverage=coverage,
                        metrics=metrics,
                    ),
                )
            )

        if not scored:
            break

        chosen_idx, probs = softmax_sample(scored, temperature=temperature, rng=rng)
        chosen = shuffled[chosen_idx]
        selected.append(chosen)
        selected_sigs.add(group_signature(chosen))

        labels = labels_by_metric(chosen, metrics)
        for metric, value in labels.items():
            coverage[(metric, metric_bin(value))] += 1

        selection_trace.append(
            {
                "signature": group_signature(chosen),
                "balance_score": dict(scored)[chosen_idx],
                "sampling_probability": probs[chosen_idx],
                "candidate_pool_size": len(scored),
            }
        )

    return selected, selection_trace


@dataclass
class BalancedGoldConfig:
    bs_sample_from: str = ""
    infer_path: str = ""
    output_path: str = ""
    metrics: str = "[]"
    shot_num: int = 20
    bootstrap_num: int = 1
    strategy: str = "score_balanced"
    agreement_raw_path: str = "datasets/mediqa-eval-2026-valid.csv"
    agreement_threshold: float = 0.0
    en_only: bool = True
    seed: int = 114514
    sample_n: int | None = None
    softmax_temperature: float = 1.0


@draccus.wrap()
def main(cfg: BalancedGoldConfig) -> None:
    metrics = ast.literal_eval(cfg.metrics)
    shot_df = pd.read_csv(cfg.bs_sample_from)
    infer_df = pd.read_csv(cfg.infer_path)

    if cfg.en_only:
        shot_df = shot_df[shot_df["lang"] == "en"].copy()
        infer_df = infer_df[infer_df["lang"] == "en"].copy()

    if cfg.strategy == "agreement_score_balanced":
        keep_keys = agreement_filter_keys(
            raw_path=cfg.agreement_raw_path,
            metrics=metrics,
            threshold=cfg.agreement_threshold,
        )
        before = len(group_examples(shot_df))
        shot_df = shot_df[shot_df.apply(lambda row: tuple(row[col] for col in KEY_COLS) in keep_keys, axis=1)].copy()
        after = len(group_examples(shot_df))
        print(f"Agreement-aware filter kept {after}/{before} shot groups", flush=True)
        if after < cfg.shot_num:
            raise ValueError(f"Agreement-aware pool too small: {after} groups for shot_num={cfg.shot_num}")
    elif cfg.strategy not in {"score_balanced", "score_balanced_softmax"}:
        raise ValueError(f"Unsupported strategy: {cfg.strategy}")

    if cfg.sample_n:
        infer_df = infer_df.sample(cfg.sample_n, random_state=cfg.seed, axis=0)

    shot_groups = group_examples(shot_df)
    infer_grouped = grouping(df=infer_df, group_type=["metric"])
    llm_input = []

    for bootstrap_id in tqdm(range(cfg.bootstrap_num)):
        rng = random.Random(cfg.seed + bootstrap_id * 9176)
        selection_trace_by_sig = {}
        if cfg.strategy == "score_balanced_softmax":
            selected_groups, selection_trace = select_score_balanced_softmax(
                groups=shot_groups,
                metrics=metrics,
                shot_num=cfg.shot_num,
                rng=rng,
                temperature=cfg.softmax_temperature,
            )
            selection_trace_by_sig = {tuple(item["signature"]): item for item in selection_trace}
        else:
            selected_groups = select_score_balanced(
                groups=shot_groups,
                metrics=metrics,
                shot_num=cfg.shot_num,
                rng=rng,
            )
        selected_df = pd.concat(selected_groups, ignore_index=True) if selected_groups else shot_df.iloc[0:0].copy()
        selected_df["bootstrap_id"] = bootstrap_id

        metadata = []
        for rank, group in enumerate(selected_groups, start=1):
            labels = labels_by_metric(group, metrics)
            trace = selection_trace_by_sig.get(group_signature(group), {})
            metadata.append(
                {
                    "dataset": str(group["dataset"].iloc[0]),
                    "encounter_id": str(group["encounter_id"].iloc[0]),
                    "candidate_author_id": str(group["candidate_author_id"].iloc[0]),
                    "rank": rank,
                    "bootstrap_id": bootstrap_id,
                    "label_bins": {metric: metric_bin(value) for metric, value in labels.items()},
                    "labels": labels,
                    "balance_score": trace.get("balance_score"),
                    "sampling_probability": trace.get("sampling_probability"),
                    "candidate_pool_size": trace.get("candidate_pool_size"),
                }
            )

        for _, infer in infer_grouped:
            shot = prepare_gold_texts_shot(
                infer_df=infer,
                shot_df=selected_df,
                metrics=metrics,
            )
            key = make_identity(infer)
            key[0]["bootstrap_id"] = bootstrap_id
            key[0]["selection_strategy"] = cfg.strategy
            shot["key"] = key
            shot["shot_metadata"] = metadata
            llm_input.append(shot)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(llm_input)} prompt items to {cfg.output_path}", flush=True)


if __name__ == "__main__":
    main()
