#!/usr/bin/env python
"""Inject lightweight retrieved guideline evidence without requiring LlamaIndex."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.embedding.hf_mean_pool import HFCLSPoolTextEncoder, HFMeanPoolTextEncoder

RAG_MARKER = "[相关临床指南]"
RAG_INSTRUCTION = (
    "以下检索到的临床指南仅作为辅助背景证据，可能不完整或只与问题部分相关。"
    "评分时应优先依据患者问题和候选回答本身。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--docstore_json", default="RAG/storage/docstore.json")
    parser.add_argument("--cache_prefix", default="RAG/simple_coder_all")
    parser.add_argument("--embed_model", default="GanjinZero/coder_all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding_backend", choices=["hf_cls_pool", "hf_mean_pool"], default="hf_cls_pool")
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--top_n", type=int, default=2)
    parser.add_argument("--max_evidence_chars", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow_duplicate", action="store_true")
    return parser.parse_args()


def flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "content" in block:
                    parts.append(str(block.get("content", "")))
        return "\n".join(parts)
    return str(content)


def set_content(item: dict, text: str) -> None:
    content = item.get("content")
    if isinstance(content, str):
        item["content"] = text
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = text
                return
    item["content"] = text


def target_start(text: str) -> int:
    marker = "The sample you need to rate:"
    idx = text.rfind(marker)
    return idx if idx >= 0 else 0


def find_query(text: str) -> str:
    target = text[target_start(text) :]
    match = re.search(r'Query:\s*"(.*?)"', target, flags=re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def find_insert_pos(text: str) -> int:
    start = target_start(text)
    llm_idx = text.find("LLM response:", start)
    if llm_idx < 0:
        return len(text)
    next_gold = text.find("\ngold responses:", llm_idx)
    next_ratings = text.find("\nRatings:", llm_idx)
    candidates = [pos for pos in [next_gold, next_ratings] if pos >= 0]
    return min(candidates) if candidates else len(text)


def load_doc_nodes(docstore_json: str) -> list[dict]:
    data = json.load(open(docstore_json, encoding="utf-8"))["docstore/data"]
    nodes = []
    seen = set()
    for node in data.values():
        raw = node.get("__data__", {})
        text = " ".join(str(raw.get("text", "")).split())
        if len(text) < 80:
            continue
        metadata = raw.get("metadata", {}) or {}
        key = (metadata.get("file_name", ""), metadata.get("page_label", ""), text[:120])
        if key in seen:
            continue
        seen.add(key)
        nodes.append(
            {
                "text": text,
                "file_name": metadata.get("file_name", "unknown"),
                "page": metadata.get("page_label") or metadata.get("page_number") or "",
            }
        )
    return nodes


def make_encoder(args: argparse.Namespace):
    cls = HFCLSPoolTextEncoder if args.embedding_backend == "hf_cls_pool" else HFMeanPoolTextEncoder
    return cls(args.embed_model, device=args.device, max_length=args.embedding_max_length)


def load_or_build_doc_embeddings(args: argparse.Namespace, encoder, nodes: list[dict]) -> np.ndarray:
    cache_prefix = Path(args.cache_prefix)
    emb_path = cache_prefix.with_suffix(".emb.npy")
    meta_path = cache_prefix.with_suffix(".meta.json")
    if emb_path.exists() and meta_path.exists():
        cached_meta = json.load(open(meta_path, encoding="utf-8"))
        if cached_meta.get("count") == len(nodes) and cached_meta.get("embed_model") == args.embed_model:
            return np.load(emb_path)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    doc_emb = encoder.encode(
        [node["text"] for node in nodes],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(emb_path, doc_emb)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(nodes),
                "embed_model": args.embed_model,
                "embedding_backend": args.embedding_backend,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return doc_emb


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + " ..."


def format_evidence(nodes: list[dict], scores: np.ndarray, idx: np.ndarray, max_chars: int, top_n: int) -> str:
    lines = []
    for rank, doc_idx in enumerate(idx[:top_n], start=1):
        node = nodes[int(doc_idx)]
        score = float(scores[int(doc_idx)])
        source = node["file_name"]
        page = f", page={node['page']}" if node.get("page") else ""
        lines.append(f"{rank}. 来源：{source}{page}，相关性={score:.4f}。{truncate(node['text'], max_chars)}")
    return f"\n\n{RAG_MARKER}:\n{RAG_INSTRUCTION}\n" + "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    items = json.load(open(args.input_json, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    nodes = load_doc_nodes(args.docstore_json)
    print(f"[SIMPLE RAG] Loaded {len(nodes)} guideline nodes")
    encoder = make_encoder(args)
    doc_emb = load_or_build_doc_embeddings(args, encoder, nodes)

    queries = [find_query(flatten_content(item.get("content", ""))) for item in items]
    query_emb = encoder.encode(
        queries,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    sims = query_emb @ doc_emb.T

    enhanced = 0
    for i, item in enumerate(items):
        text = flatten_content(item.get("content", ""))
        if RAG_MARKER in text and not args.allow_duplicate:
            continue
        if not queries[i]:
            continue
        top_idx = np.argsort(-sims[i])[: args.retrieve_top_k]
        evidence = format_evidence(nodes, sims[i], top_idx, args.max_evidence_chars, args.top_n)
        insert_pos = find_insert_pos(text)
        set_content(item, text[:insert_pos] + evidence + text[insert_pos:])
        enhanced += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"[SIMPLE RAG] Enhanced {enhanced}/{len(items)} prompts")
    print(f"[SIMPLE RAG] Saved {output_path}")


if __name__ == "__main__":
    main()
