import ast
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import draccus
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp.few_shot.scripts.make_shot import make_identity
from exp.few_shot.scripts.prompts import gold_text_template, prompt_outputTemp
from utils.dataset_helper.get_data import grouping_with_gold


def format_gold_texts(raw_gold_texts: str) -> str:
    if not isinstance(raw_gold_texts, str) or not raw_gold_texts.strip():
        return "(no gold responses available)"

    cleaned = raw_gold_texts.strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            parsed = ast.literal_eval(cleaned)
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        items = [str(item).strip() for item in parsed if str(item).strip()]
        if items:
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))

    return cleaned


def build_rag_prompt(infer_df: pd.DataFrame, metrics: list[str]) -> dict:
    base_prompt = gold_text_template(metrics=metrics)
    grouped_infer_df = grouping_with_gold(infer_df, group_type=["metric"])

    prompt = base_prompt + f"""

Now rate the response below. Remember to output only JSON.
Do not add Markdown formatting, backticks, or explanations.
The output must be valid JSON and strictly follow this schema:
{prompt_outputTemp(metrics=metrics)}
"""

    for _, infer_group in grouped_infer_df:
        infer_query = infer_group["query_text"].iloc[0]
        infer_response = infer_group["candidate"].iloc[0]
        infer_gold_response = format_gold_texts(infer_group["gold_texts"].iloc[0])

        prompt += f"""

The sample you need to rate:
Query:
"{infer_query}"

LLM response:
"{infer_response}"

Gold doctor responses:
{infer_gold_response}
"""

    return {
        "role": "user",
        "content": prompt,
        "key": None,
    }


@dataclass
class config:
    infer_path: str = "datasets/mediqa-eval-2026-valid-aligned.csv"
    output_path: str = "results/rag_input.json"
    metrics: str = "['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"
    sample_n: int | None = None
    lang: str = "en"


@draccus.wrap()
def main(cfg: config):
    metrics = ast.literal_eval(cfg.metrics)
    infer_df = pd.read_csv(cfg.infer_path)

    if cfg.lang and cfg.lang.lower() != "all":
        infer_df = infer_df[infer_df["lang"] == cfg.lang.lower()].copy()

    if metrics:
        infer_df = infer_df[infer_df["metric"].isin(metrics)].copy()

    infer_groups = list(grouping_with_gold(infer_df, group_type=["metric"]))
    if cfg.sample_n:
        infer_groups = infer_groups[: cfg.sample_n]

    llm_input = []
    for _, infer_group in infer_groups:
        prompt = build_rag_prompt(infer_df=infer_group, metrics=metrics)
        prompt["key"] = make_identity(infer_group)
        llm_input.append(prompt)

    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(llm_input)} prompts to {cfg.output_path}")
    if llm_input:
        print(f"First prompt key: {llm_input[0]['key']}")


if __name__ == "__main__":
    main()
