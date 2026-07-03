#!/usr/bin/env python
"""Inject retrieved clinical guideline evidence into MEDIQA LLM input JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Iterable

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.embedding.hf_mean_pool import LlamaIndexHFMeanPoolEmbedding

RAG_MARKER = "[Relevant Clinical Guidelines]"
RAG_INSTRUCTION = (
    "Use the retrieved clinical guidelines below only as auxiliary background evidence. "
    "They may be incomplete or only partially relevant. Do not penalize the candidate response "
    "for omitting guideline details that are not directly required by the patient's question. "
    "Prioritize the patient query and candidate response when assigning all ratings."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--persist_dir", default="RAG/storage")
    parser.add_argument("--embed_model", default="/data/public_models/bge-m3")
    parser.add_argument(
        "--embedding_backend",
        choices=["huggingface", "hf_mean_pool"],
        default="huggingface",
    )
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--rerank_model", default="/data/public_models/bge-reranker-large")
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--rerank_top_n", type=int, default=2)
    parser.add_argument("--disable_rerank", action="store_true")
    parser.add_argument("--max_evidence_chars", type=int, default=1200)
    parser.add_argument(
        "--query_source",
        choices=["target_query", "gold_responses", "query_plus_gold"],
        default="target_query",
        help="Text used as the retrieval query.",
    )
    parser.add_argument(
        "--insert_location",
        choices=["auto", "after_query", "after_gold", "after_llm_response", "before_llm_response", "end"],
        default="auto",
    )
    parser.add_argument("--include_metadata", action="store_true")
    parser.add_argument("--allow_duplicate", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records.")
    return parser.parse_args()


def load_resources(args: argparse.Namespace):
    print(f"[RAG] Loading embedder: {args.embed_model}")
    if args.embedding_backend == "huggingface":
        Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    elif args.embedding_backend == "hf_mean_pool":
        Settings.embed_model = LlamaIndexHFMeanPoolEmbedding.create(
            model_name=args.embed_model,
            max_length=args.embedding_max_length,
        )
    else:
        raise ValueError(f"Unsupported embedding_backend: {args.embedding_backend}")

    reranker = None
    if args.disable_rerank:
        print("[RAG] Reranker disabled")
    else:
        print(f"[RAG] Loading reranker: {args.rerank_model}")
        reranker = SentenceTransformerRerank(
            model=args.rerank_model,
            top_n=args.rerank_top_n,
        )

    print(f"[RAG] Loading index from: {args.persist_dir}")
    storage_context = StorageContext.from_defaults(persist_dir=args.persist_dir)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=args.retrieve_top_k)
    return retriever, reranker


def iter_text_blocks(item: dict) -> Iterable[tuple[dict, str, str]]:
    content = item.get("content")
    if isinstance(content, str):
        yield item, "content", content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    yield block, "text", text


def target_start(text: str) -> int:
    marker = "The sample you need to rate:"
    idx = text.rfind(marker)
    return idx if idx >= 0 else 0


def find_query(text: str) -> re.Match[str] | None:
    start = target_start(text)
    return re.search(r'Query:\s*"(.*?)"', text[start:], re.DOTALL)


def find_gold_block(text: str) -> re.Match[str] | None:
    start = target_start(text)
    target = text[start:]
    pattern = re.compile(
        r"(?:Gold doctor responses|gold responses):\s*(.*?)(?=\n\s*\n(?:Now rate|The sample|$)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    return pattern.search(target)


def absolute_span(text: str, match: re.Match[str]) -> tuple[int, int]:
    start = target_start(text)
    return start + match.start(), start + match.end()


def parse_gold_text(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return " ".join(str(x).strip() for x in parsed if str(x).strip())
        except Exception:
            pass

    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s*\d+\.\s*", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def get_retrieval_query(text: str, query_source: str) -> str | None:
    query_text = None
    query_match = find_query(text)
    if query_match:
        query_text = query_match.group(1).strip()

    gold_text = None
    gold_match = find_gold_block(text)
    if gold_match:
        gold_text = parse_gold_text(gold_match.group(1))

    if query_source == "target_query":
        return query_text
    if query_source == "gold_responses":
        return gold_text
    if query_text and gold_text:
        return f"{query_text} {gold_text}"
    return query_text or gold_text


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated} ..."


def format_evidence(query_text: str, retriever, reranker, include_metadata: bool, max_evidence_chars: int, top_n: int) -> str | None:
    if not query_text or len(query_text.strip()) < 5:
        return None

    nodes = retriever.retrieve(query_text)
    if reranker is not None:
        nodes = reranker.postprocess_nodes(nodes, query_str=query_text)
    else:
        nodes = nodes[:top_n]

    evidence = []
    for idx, node in enumerate(nodes, start=1):
        text = " ".join(node.node.get_text().split())
        if not text:
            continue
        text = truncate_text(text, max_evidence_chars)
        if include_metadata:
            metadata = node.node.metadata or {}
            source = metadata.get("file_name") or metadata.get("file_path") or "unknown"
            page = metadata.get("page_label") or metadata.get("page_number")
            score = getattr(node, "score", None)
            score_text = f", score={score:.4f}" if isinstance(score, float) else ""
            page_text = f", page={page}" if page is not None else ""
            evidence.append(f"{idx}. Source: {source}{page_text}{score_text}. {text}")
        else:
            evidence.append(f"{idx}. {text}")

    if not evidence:
        return None
    return f"\n\n{RAG_MARKER}:\n{RAG_INSTRUCTION}\n" + "\n".join(evidence) + "\n"


def find_llm_response_block_end(text: str) -> int | None:
    start = target_start(text)
    idx = text.find("LLM response:", start)
    if idx < 0:
        return None
    ratings_idx = text.find("\nRatings:", idx)
    next_sample_idx = text.find("\nThe sample you need to rate:", idx + len("LLM response:"))
    candidates = [pos for pos in [ratings_idx, next_sample_idx] if pos >= 0]
    return min(candidates) if candidates else len(text)


def choose_insert_pos(text: str, location: str, query_source: str) -> int:
    if location == "auto":
        location = "after_gold" if query_source == "gold_responses" else "after_llm_response"

    if location == "after_query":
        match = find_query(text)
        if match:
            return absolute_span(text, match)[1]

    if location == "after_gold":
        match = find_gold_block(text)
        if match:
            return absolute_span(text, match)[1]

    if location == "before_llm_response":
        start = target_start(text)
        idx = text.find("LLM response:", start)
        if idx >= 0:
            return idx

    if location == "after_llm_response":
        idx = find_llm_response_block_end(text)
        if idx is not None:
            return idx

    return len(text)


def process_text(text: str, args: argparse.Namespace, retriever, reranker) -> tuple[str, bool]:
    if RAG_MARKER in text and not args.allow_duplicate:
        return text, False

    query_text = get_retrieval_query(text, args.query_source)
    evidence = format_evidence(
        query_text or "",
        retriever,
        reranker,
        args.include_metadata,
        args.max_evidence_chars,
        args.rerank_top_n,
    )
    if not evidence:
        return text, False

    insert_pos = choose_insert_pos(text, args.insert_location, args.query_source)
    return text[:insert_pos] + evidence + text[insert_pos:], True


def main() -> None:
    args = parse_args()
    retriever, reranker = load_resources(args)

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[: args.limit]

    processed = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        for container, key, text in iter_text_blocks(item):
            new_text, changed = process_text(text, args, retriever, reranker)
            if changed:
                container[key] = new_text
                processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[RAG] Enhanced {processed} text blocks")
    print(f"[RAG] Saved output to {output_path}")


if __name__ == "__main__":
    main()
