#!/usr/bin/env python
"""Convert MEDIQA reference-based few-shot prompts into no-gold prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GOLD_MARKERS = ("gold responses", "Gold doctor responses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--assert_no_gold", action="store_true")
    return parser.parse_args()


def strip_gold_instructions(text: str) -> str:
    replacements = [
        (
            "You will be supplied with gold responses given by professional doctors.\n"
            "    You should rate the LLM's response by comparing it very carefully with the gold responses and treat them as ground truth.",
            "You should rate the LLM's response using the patient query, candidate answer, and established dermatology/wound-care medical knowledge.",
        ),
        (
            "You will be supplied with gold responses given by professional doctors.",
            "Use established dermatology/wound-care medical knowledge when judging the response.",
        ),
        (
            "You should rate the LLM's response by comparing it very carefully with the gold responses and treat them as ground truth.",
            "You should rate the LLM's response using the patient query, candidate answer, and established medical knowledge.",
        ),
        (
            "which contains only a query, a response and gold responses.",
            "which contains only a query and a response.",
        ),
        (
            "which contains only a query, a response, and gold responses.",
            "which contains only a query and a response.",
        ),
        (
            "factual_consistency_wgold:compare the candidate response against the provided gold doctor responses. "
            "Put 1 if it is fully consistent, 0.5 if partially consistent or missing key details, 0 if inconsistent or medically wrong.",
            "factual_consistency_wgold:judge whether the candidate response is medically correct based on the patient query and established medical knowledge. "
            "Put 1 if it is fully correct, 0.5 if partially correct or missing key details, 0 if medically wrong.",
        ),
        (
            "provided gold doctor responses",
            "the patient query and established medical knowledge",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def strip_gold_blocks(text: str) -> str:
    # Few-shot examples: keep Ratings after removing the gold-response block.
    text = re.sub(
        r"(?im)^(?:Gold doctor responses|gold responses):.*?(?=^Ratings:)",
        "\n",
        text,
        flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )

    # Target sample: gold-response block usually appears after LLM response and before
    # the next blank section or the end of the prompt.
    text = re.sub(
        r"(?im)^(?:Gold doctor responses|gold responses):.*?(?=^\s*(?:\[Relevant Clinical Guidelines|Now rate|The sample you need to rate:|Sample \d+:)|\Z)",
        "\n",
        text,
        flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    return text


def convert_text(text: str) -> str:
    text = strip_gold_instructions(text)
    text = strip_gold_blocks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def iter_text_blocks(item: dict):
    content = item.get("content")
    if isinstance(content, str):
        yield item, "content", content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text", "")
                if isinstance(value, str):
                    yield block, "text", value


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    converted_blocks = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        for container, key, text in iter_text_blocks(item):
            new_text = convert_text(text)
            if new_text != text:
                converted_blocks += 1
                container[key] = new_text

    if args.assert_no_gold:
        joined = json.dumps(data, ensure_ascii=False)
        found = [marker for marker in GOLD_MARKERS if marker.lower() in joined.lower()]
        if found:
            raise SystemExit(f"Gold-response marker still present after conversion: {found}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Converted {converted_blocks} text blocks")
    print(f"Saved no-gold input to {output_path}")


if __name__ == "__main__":
    main()
