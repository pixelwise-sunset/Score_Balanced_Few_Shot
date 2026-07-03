#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_tables_30b_shot20_pubmedbert}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/public_models/bert/pubmedbert}"
export EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-hf_mean_pool}"
export EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-512}"
export RAG_PERSIST_DIR="${RAG_PERSIST_DIR:-RAG/storage_pubmedbert}"
export REBUILD_RAG_INDEX="${REBUILD_RAG_INDEX:-1}"
export DISABLE_RERANK="${DISABLE_RERANK:-1}"

export WITH_GOLD_RUN_ID="${WITH_GOLD_RUN_ID:-main30b-shot20-with-gold-pubmedbert-mmr-oof-singleload}"
export NO_GOLD_RUN_ID="${NO_GOLD_RUN_ID:-main30b-shot20-no-gold-pubmedbert-mmr-oof-singleload}"
export RAG_RUN_ID="${RAG_RUN_ID:-main30b-shot20-no-gold-rag-v2-pubmedbert-query-mmr-oof-singleload}"

bash scripts/run_main_tables_30b_shot20.sh
