#!/usr/bin/env python
"""Build a LlamaIndex vector index from the MEDIQA clinical guideline PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.embedding.hf_mean_pool import LlamaIndexHFMeanPoolEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf_dir", default="RAG/RAG_files")
    parser.add_argument("--persist_dir", default="RAG/storage")
    parser.add_argument("--embed_model", default="/data/public_models/bge-m3")
    parser.add_argument(
        "--embedding_backend",
        choices=["huggingface", "hf_mean_pool"],
        default="huggingface",
    )
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    persist_dir = Path(args.persist_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    if args.embedding_backend == "huggingface":
        Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    elif args.embedding_backend == "hf_mean_pool":
        Settings.embed_model = LlamaIndexHFMeanPoolEmbedding.create(
            model_name=args.embed_model,
            max_length=args.embedding_max_length,
        )
    else:
        raise ValueError(f"Unsupported embedding_backend: {args.embedding_backend}")
    Settings.node_parser = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"[RAG] Loading documents from {pdf_dir}")
    documents = SimpleDirectoryReader(str(pdf_dir), recursive=True).load_data()
    print(f"[RAG] Loaded {len(documents)} document pages/chunks")

    print("[RAG] Building vector index")
    index = VectorStoreIndex.from_documents(documents)

    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"[RAG] Saved index to {persist_dir}")


if __name__ == "__main__":
    main()
