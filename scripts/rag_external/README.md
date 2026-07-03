# MEDIQA External-Knowledge RAG

This folder converts the original `RAG/BuildRAG.ipynb` workflow into command-line scripts.

The migrated RAG workflow:

1. Build or load a LlamaIndex vector index from clinical guideline PDFs under `RAG/RAG_files`.
2. Embed queries with `bge-m3`.
3. Retrieve top-k chunks from the guideline index.
4. Rerank retrieved chunks with `bge-reranker-large`.
5. Inject the selected evidence into MEDIQA model-input JSON as `[Relevant Clinical Guidelines (RAG Prediction)]`.

Example:

```bash
cd /home/liyuan/projects/mediqa

/data/kunfeng/miniconda3/envs/qwen3/bin/python scripts/rag_external/augment_rag_input.py \
  --input_json results/rag_gold_qwen3_8b_repro_20260602_011341_ALL/rag_input.json \
  --output_json results/rag_external_smoke/rag_input_external.json \
  --persist_dir RAG/storage \
  --embed_model /data/public_models/bge-m3 \
  --rerank_model /data/public_models/bge-reranker-large \
  --query_source target_query \
  --include_metadata
```

`--query_source target_query` is the cleaner paper baseline because it retrieves from the patient query only.
`--query_source gold_responses` preserves the notebook variant that retrieves using the target sample's doctor responses.
