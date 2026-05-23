#!/bin/bash
set -e
COLLECTION="${1:-documents}"
TOP_K="${2:-5}"
USE_RERANK="${3:-false}"

cd "$(dirname "$0")/.."
echo "Starting question and answer loop..."

# Build command with optional reranking flag
CMD=".venv/bin/python src/query.py --collection $COLLECTION --top-k $TOP_K --show-chunks"
if [[ "$USE_RERANK" == "true" ]] || [[ "$USE_RERANK" == "1" ]]; then
  CMD="$CMD --rerank"
  echo "Reranking enabled (retrieves top-20, reranks to top-$TOP_K)"
fi

$CMD
