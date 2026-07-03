import ast
import json
import random
from dataclasses import dataclass

import draccus
import pandas as pd
from tqdm import tqdm

from exp.few_shot.scripts.make_shot import make_identity, prepare_gold_texts_shot
from utils.dataset_helper.get_data import grouping


def sample_groups(df: pd.DataFrame, sample_n: int, rng: random.Random, replace: bool) -> pd.DataFrame:
    if sample_n <= 0:
        return df.iloc[0:0].copy()

    grouped_df = grouping(df, group_type=["metric"])
    keys = list(grouped_df.groups.keys())
    if replace:
        chosen_keys = rng.choices(keys, k=sample_n)
    else:
        chosen_keys = rng.sample(keys, sample_n)

    sampled_groups = []
    for copy_id, key in enumerate(chosen_keys):
        group = grouped_df.get_group(key).copy()
        group["bootstrap_copy_id"] = copy_id
        sampled_groups.append(group)

    return pd.concat(sampled_groups, ignore_index=True)


@dataclass
class BootstrapGoldConfig:
    bs_sample_from: str = ""
    infer_path: str = ""
    output_path: str = ""
    metrics: str = "[]"
    shot_num: int = 20
    bootstrap_num: int = 7
    sample_n: int = None
    en_only: bool = True
    sample_with_replacement: bool = True
    seed: int = 114514


@draccus.wrap()
def main(cfg: BootstrapGoldConfig):
    metrics = ast.literal_eval(cfg.metrics)
    rng = random.Random(cfg.seed)

    shot_df = pd.read_csv(cfg.bs_sample_from)
    infer_df = pd.read_csv(cfg.infer_path)

    if cfg.en_only:
        shot_df = shot_df[shot_df["lang"] == "en"].copy()
        infer_df = infer_df[infer_df["lang"] == "en"].copy()

    if cfg.sample_n:
        infer_df = infer_df.sample(cfg.sample_n, random_state=cfg.seed, axis=0)

    llm_input = []
    infer_group = grouping(df=infer_df, group_type=["metric"])

    for bootstrap_id in tqdm(range(cfg.bootstrap_num)):
        shot_sample = sample_groups(
            df=shot_df,
            sample_n=cfg.shot_num,
            rng=rng,
            replace=cfg.sample_with_replacement,
        )
        shot_sample["bootstrap_id"] = bootstrap_id

        for _, infer in infer_group:
            shot = prepare_gold_texts_shot(
                infer_df=infer,
                shot_df=shot_sample,
                metrics=metrics,
            )
            key = make_identity(infer)
            key[0]["bootstrap_id"] = bootstrap_id
            shot["key"] = key
            llm_input.append(shot)

    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(llm_input)} prompt items to {cfg.output_path}")


if __name__ == "__main__":
    main()
