import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import draccus
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp.few_shot.scripts.make_shot import make_identity, prepare_gold_texts_shot
from exp.few_shot.scripts.prompts import gold_text_template, prompt_outputTemp
from utils.embedding.hf_mean_pool import HFCLSPoolTextEncoder, HFMeanPoolTextEncoder
from utils.dataset_helper.get_data import grouping, grouping_with_gold


def group_samples(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [group.copy() for _, group in grouping_with_gold(df, group_type=["metric"])]


def group_infer_samples(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [group.copy() for _, group in grouping(df, group_type=["metric"])]


def sample_text(group: pd.DataFrame) -> str:
    row = group.iloc[0]
    parts = [
        f"Query: {row.get('query_text', '')}",
        f"LLM response: {row.get('candidate', '')}",
        f"Gold doctor responses: {row.get('gold_texts', '')}",
    ]
    image_caption = row.get("image_caption", "")
    if isinstance(image_caption, str) and image_caption.strip():
        parts.append(f"Image caption: {image_caption}")
    return "\n".join(parts)


def normalize_weights(similarities: np.ndarray) -> np.ndarray:
    weights = np.maximum(similarities.astype(float), 0.0)
    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights, dtype=float) / len(weights)
    return weights / total


def select_topk(similarities: np.ndarray, shot_num: int) -> np.ndarray:
    top_k = min(shot_num, len(similarities))
    return np.argsort(-similarities)[:top_k]


def select_similarity_sample(
    similarities: np.ndarray,
    shot_num: int,
    pool_multiplier: int,
    rng: np.random.Generator,
) -> np.ndarray:
    top_k = min(shot_num, len(similarities))
    if top_k <= 0:
        return np.asarray([], dtype=int)

    pool_size = min(len(similarities), max(top_k, top_k * pool_multiplier))
    candidate_idx = np.argsort(-similarities)[:pool_size]
    weights = normalize_weights(similarities[candidate_idx])
    replace = pool_size < top_k
    return rng.choice(candidate_idx, size=top_k, replace=replace, p=weights)


def select_mmr(
    similarities: np.ndarray,
    shot_emb: np.ndarray,
    shot_num: int,
    pool_multiplier: int,
    mmr_lambda: float,
    initial_idx: int | None = None,
) -> np.ndarray:
    top_k = min(shot_num, len(similarities))
    if top_k <= 0:
        return np.asarray([], dtype=int)

    pool_size = min(len(similarities), max(top_k, top_k * pool_multiplier))
    candidates = list(np.argsort(-similarities)[:pool_size])
    if initial_idx is not None and initial_idx in candidates:
        selected = [initial_idx]
        candidates.remove(initial_idx)
    else:
        selected = [candidates.pop(0)]

    if not candidates:
        return np.asarray(selected, dtype=int)

    while candidates and len(selected) < top_k:
        candidate_arr = np.asarray(candidates, dtype=int)
        selected_arr = np.asarray(selected, dtype=int)
        redundancy = shot_emb[candidate_arr] @ shot_emb[selected_arr].T
        max_redundancy = redundancy.max(axis=1)
        mmr_scores = mmr_lambda * similarities[candidate_arr] - (1.0 - mmr_lambda) * max_redundancy
        best_pos = int(np.argmax(mmr_scores))
        selected.append(candidates.pop(best_pos))

    return np.asarray(selected, dtype=int)


def select_mmr_sample(
    similarities: np.ndarray,
    shot_emb: np.ndarray,
    shot_num: int,
    pool_multiplier: int,
    mmr_lambda: float,
    rng: np.random.Generator,
) -> np.ndarray:
    top_k = min(shot_num, len(similarities))
    if top_k <= 0:
        return np.asarray([], dtype=int)

    pool_size = min(len(similarities), max(top_k, top_k * pool_multiplier))
    candidate_idx = np.argsort(-similarities)[:pool_size]
    warm_start_size = min(pool_size, max(1, min(top_k, 20)))
    warm_start_idx = candidate_idx[:warm_start_size]
    weights = normalize_weights(similarities[warm_start_idx])
    initial_idx = int(rng.choice(warm_start_idx, p=weights))
    return select_mmr(
        similarities=similarities,
        shot_emb=shot_emb,
        shot_num=shot_num,
        pool_multiplier=pool_multiplier,
        mmr_lambda=mmr_lambda,
        initial_idx=initial_idx,
    )


def prepare_weighted_gold_texts_shot(
    infer_df: pd.DataFrame,
    shot_df: pd.DataFrame,
    metrics: list[str],
    similarities: list[float],
    weights: list[float],
):
    base_prompt = gold_text_template(metrics=metrics)

    shot_groups = [group.copy() for _, group in grouping_with_gold(shot_df, group_type=["metric"])]
    infer_groups = [group.copy() for _, group in grouping_with_gold(infer_df, group_type=["metric"])]

    for sample_idx, shot_group in enumerate(shot_groups):
        shot_query = shot_group["query_text"].iloc[0]
        shot_response = shot_group["candidate"].iloc[0]
        shot_gold_response = str(shot_group["gold_texts"].iloc[0])

        rating_string = ""
        for metric in metrics:
            metric_value = shot_group["label"][shot_group["metric"] == metric].iloc[0]
            rating_string += f"{metric}: {metric_value}\n"

        example_string = (
            f"Sample {sample_idx + 1}:\n"
            f"Similarity to target: {similarities[sample_idx]:.6f}\n"
            f"Normalized example weight: {weights[sample_idx]:.6f}\n"
            "Use higher-weight examples as more relevant calibration examples.\n"
            f"Query:\n\"{shot_query}\"\n\n"
            f"LLM response:\n\"{shot_response}\"\n"
            f"gold responses:{shot_gold_response}\n"
            f"Ratings:\n{rating_string}"
        )
        base_prompt += "\n\n" + example_string

    base_prompt += f"""
        Now rate the response below. Remember to output only JSON.
        Do **not** add any Markdown formatting, backticks, or explanations.
        The output must be a valid JSON object or array only.
        Your response shoud strictly follow: \n{prompt_outputTemp(metrics=metrics)}
                        """ + "\n\n"

    for infer_group in infer_groups:
        infer_query = infer_group["query_text"].iloc[0]
        infer_response = infer_group["candidate"].iloc[0]
        infer_gold_response = str(infer_group["gold_texts"].iloc[0])
        base_prompt += (
            "The sample you need to rate:\n"
            f"Query:\n\"{infer_query}\"\n\n"
            f"LLM response:\n\"{infer_response}\"\n"
            f"gold responses:{infer_gold_response}\n"
        )

    return {"role": "user", "content": base_prompt, "key": None}


@dataclass
class SimilarityGoldConfig:
    bs_sample_from: str = ""
    infer_path: str = ""
    output_path: str = ""
    metrics: str = "[]"
    shot_num: int = 20
    embedding_model: str = "/data/public_models/bge-m3"
    embedding_backend: str = "sentence_transformer"
    embedding_max_length: int = 512
    strategy: str = "similarity_topk"
    en_only: bool = True
    sample_n: int | None = None
    bootstrap_num: int = 1
    seed: int = 114514
    batch_size: int = 32
    pool_multiplier: int = 5
    mmr_lambda: float = 0.7


@draccus.wrap()
def main(cfg: SimilarityGoldConfig):
    metrics = ast.literal_eval(cfg.metrics)
    shot_df = pd.read_csv(cfg.bs_sample_from)
    infer_df = pd.read_csv(cfg.infer_path)

    if cfg.en_only:
        shot_df = shot_df[shot_df["lang"] == "en"].copy()
        infer_df = infer_df[infer_df["lang"] == "en"].copy()

    if cfg.sample_n:
        infer_df = infer_df.sample(cfg.sample_n, random_state=cfg.seed, axis=0)

    shot_groups = group_samples(shot_df)
    infer_groups = group_infer_samples(infer_df)

    if cfg.embedding_backend == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg.embedding_model)
    elif cfg.embedding_backend == "hf_mean_pool":
        model = HFMeanPoolTextEncoder(
            cfg.embedding_model,
            max_length=cfg.embedding_max_length,
        )
    elif cfg.embedding_backend == "hf_cls_pool":
        model = HFCLSPoolTextEncoder(
            cfg.embedding_model,
            max_length=cfg.embedding_max_length,
        )
    else:
        raise ValueError(f"Unsupported embedding_backend: {cfg.embedding_backend}")
    llm_input = []

    if cfg.shot_num > 0:
        shot_texts = [sample_text(group) for group in shot_groups]
        shot_emb = model.encode(
            shot_texts,
            batch_size=cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        shot_emb = np.asarray(shot_emb, dtype=np.float32)
    else:
        shot_emb = np.zeros((0, 1), dtype=np.float32)

    infer_texts = [sample_text(group) for group in infer_groups]
    infer_emb = model.encode(
        infer_texts,
        batch_size=cfg.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    infer_emb = np.asarray(infer_emb, dtype=np.float32)

    effective_bootstrap_num = 1 if cfg.shot_num <= 0 else cfg.bootstrap_num

    for infer_idx, infer_group in enumerate(tqdm(infer_groups)):
        similarities = shot_emb @ infer_emb[infer_idx] if cfg.shot_num > 0 else np.asarray([])

        for bootstrap_id in range(effective_bootstrap_num):
            if cfg.shot_num <= 0:
                selected_df = shot_df.iloc[0:0].copy()
                shot = prepare_gold_texts_shot(
                    infer_df=infer_group,
                    shot_df=selected_df,
                    metrics=metrics,
                )
                shot["shot_metadata"] = []
            else:
                rng = np.random.default_rng(cfg.seed + infer_idx * 1009 + bootstrap_id * 9176)
                if cfg.strategy == "similarity_mmr":
                    if bootstrap_id == 0:
                        top_idx = select_mmr(
                            similarities=similarities,
                            shot_emb=shot_emb,
                            shot_num=cfg.shot_num,
                            pool_multiplier=cfg.pool_multiplier,
                            mmr_lambda=cfg.mmr_lambda,
                        )
                    else:
                        top_idx = select_mmr_sample(
                            similarities=similarities,
                            shot_emb=shot_emb,
                            shot_num=cfg.shot_num,
                            pool_multiplier=cfg.pool_multiplier,
                            mmr_lambda=cfg.mmr_lambda,
                            rng=rng,
                        )
                elif bootstrap_id == 0:
                    top_idx = select_topk(similarities=similarities, shot_num=cfg.shot_num)
                else:
                    top_idx = select_similarity_sample(
                        similarities=similarities,
                        shot_num=cfg.shot_num,
                        pool_multiplier=cfg.pool_multiplier,
                        rng=rng,
                    )

                top_k = len(top_idx)
                selected_groups = [shot_groups[i].copy() for i in top_idx]
                selected_df = pd.concat(selected_groups, ignore_index=True)
                selected_sims = similarities[top_idx].astype(float)
                selected_weights = normalize_weights(selected_sims)

                if cfg.strategy == "similarity_weighted":
                    shot = prepare_weighted_gold_texts_shot(
                        infer_df=infer_group,
                        shot_df=selected_df,
                        metrics=metrics,
                        similarities=selected_sims.tolist(),
                        weights=selected_weights.tolist(),
                    )
                else:
                    shot = prepare_gold_texts_shot(
                        infer_df=infer_group,
                        shot_df=selected_df,
                        metrics=metrics,
                    )
                shot["shot_metadata"] = [
                    {
                        "dataset": selected_groups[pos]["dataset"].iloc[0],
                        "encounter_id": selected_groups[pos]["encounter_id"].iloc[0],
                        "candidate_author_id": selected_groups[pos]["candidate_author_id"].iloc[0],
                        "similarity": float(selected_sims[pos]),
                        "weight": float(selected_weights[pos]),
                        "rank": pos + 1,
                    }
                    for pos in range(top_k)
                ]

            key = make_identity(infer_group)
            key[0]["bootstrap_id"] = bootstrap_id
            key[0]["selection_strategy"] = cfg.strategy
            shot["key"] = key
            llm_input.append(shot)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(llm_input)} prompt items to {cfg.output_path}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
