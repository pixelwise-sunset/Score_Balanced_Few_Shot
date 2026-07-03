#!/usr/bin/env python
"""Build a translated docstore from one or more translation progress JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_docstore", default="RAG/storage/docstore.json")
    parser.add_argument("--output_docstore", default="RAG/storage_zh/docstore.json")
    parser.add_argument("--progress_jsonl", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docstore = json.loads(Path(args.input_docstore).read_text(encoding="utf-8"))
    translations = {}
    for progress_path in args.progress_jsonl:
        path = Path(progress_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                translations[item["node_id"]] = item["translated_text"]

    for node_id, translated_text in translations.items():
        node = docstore.get("docstore/data", {}).get(node_id)
        if node is not None:
            node.setdefault("__data__", {})["text"] = translated_text

    output_path = Path(args.output_docstore)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(docstore, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_path}")
    print(f"Translations merged: {len(translations)}")


if __name__ == "__main__":
    main()
