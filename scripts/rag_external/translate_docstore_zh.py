#!/usr/bin/env python
"""Translate a LlamaIndex docstore into Simplified Chinese for Chinese RAG."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_docstore", default="RAG/storage/docstore.json")
    parser.add_argument("--output_docstore", default="RAG/storage_zh/docstore.json")
    parser.add_argument("--progress_jsonl", default="RAG/storage_zh/translation_progress.jsonl")
    parser.add_argument("--model_path", default="/workspace/models/medgemma-1.5-4b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_input_chars", type=int, default=2400)
    parser.add_argument("--max_new_tokens", type=int, default=2300)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_progress(path: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            done[item["node_id"]] = item["translated_text"]
    return done


def get_node_text(node: dict) -> str:
    raw = node.get("__data__", {})
    return str(raw.get("text", "") or "")


def set_node_text(node: dict, text: str) -> None:
    node.setdefault("__data__", {})["text"] = text


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha() or "\u4e00" <= ch <= "\u9fff")
    return zh / max(letters, 1)


def split_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in re.split(r"(?<=[.;:])\s+", text) if p.strip()]

    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                parts.append(buf.strip())
                buf = ""
            for i in range(0, len(para), max_chars):
                parts.append(para[i : i + max_chars].strip())
            continue
        if len(buf) + len(para) + 2 > max_chars:
            parts.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        parts.append(buf.strip())
    return parts


def clean_translation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:zh|chinese)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^(好的，?)?(这是|以下是)?(您要求的)?(简体中文)?翻译[:：]\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    return text


def make_prompt(text: str, retry: bool = False) -> list[dict[str, str]]:
    extra = ""
    if retry:
        extra = "上一次输出保留了过多英文。请逐句翻译正文中的英文内容，除药名、缩写、URL和版权声明外不要复制英文原句。\n"
    return [
        {
            "role": "user",
            "content": (
                "你是医学指南翻译器。请将以下医学指南文本逐句翻译成简体中文。\n"
                "要求：只输出译文；保留医学含义、数字、药名、缩写、列表结构和引用信息；不要添加解释。\n\n"
                f"{extra}"
                f"{text}"
            ),
        }
    ]


def translate_chunk(model, tokenizer, text: str, args: argparse.Namespace, retry: bool = False) -> str:
    messages = make_prompt(text, retry=retry)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs.input_ids.shape[-1] :]
    return clean_translation(tokenizer.decode(new_tokens, skip_special_tokens=True))


def translate_text(model, tokenizer, text: str, args: argparse.Namespace) -> str:
    text = text.strip()
    if not text:
        return text
    if chinese_ratio(text) > 0.45:
        return text
    chunks = split_text(text, args.max_input_chars)
    translated = []
    for chunk in chunks:
        first = translate_chunk(model, tokenizer, chunk, args)
        if chinese_ratio(chunk) < 0.2 and chinese_ratio(first) < 0.15 and len(chunk) > 120:
            first = translate_chunk(model, tokenizer, chunk, args, retry=True)
        translated.append(first)
    return "\n\n".join(part for part in translated if part).strip()


def write_docstore(docstore: dict, translations: dict[str, str], output_path: Path) -> None:
    out = copy.deepcopy(docstore)
    for node_id, translated in translations.items():
        node = out.get("docstore/data", {}).get(node_id)
        if node is not None:
            set_node_text(node, translated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_docstore)
    output_path = Path(args.output_docstore)
    progress_path = Path(args.progress_jsonl)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    docstore = json.loads(input_path.read_text(encoding="utf-8"))
    nodes = []
    for idx, (node_id, node) in enumerate(docstore.get("docstore/data", {}).items()):
        if args.num_shards > 1 and idx % args.num_shards != args.shard_index:
            continue
        text = get_node_text(node)
        if len(text.strip()) >= 80:
            nodes.append((node_id, text))
    if args.limit:
        nodes = nodes[: args.limit]

    translations = {} if args.force else load_progress(progress_path)
    print(f"Loaded {len(nodes)} candidate nodes; already translated {len(translations)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )

    completed_since_save = 0
    with progress_path.open("a", encoding="utf-8") as progress_f:
        for node_id, text in tqdm(nodes):
            if node_id in translations and not args.force:
                continue
            translated = translate_text(model, tokenizer, text, args)
            translations[node_id] = translated
            progress_f.write(
                json.dumps(
                    {
                        "node_id": node_id,
                        "original_len": len(text),
                        "translated_len": len(translated),
                        "translated_text": translated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            progress_f.flush()
            completed_since_save += 1
            if completed_since_save >= args.save_every:
                write_docstore(docstore, translations, output_path)
                completed_since_save = 0

    write_docstore(docstore, translations, output_path)
    print(f"Saved translated docstore to {output_path}")
    print(f"Translated nodes available: {len(translations)}")


if __name__ == "__main__":
    main()
