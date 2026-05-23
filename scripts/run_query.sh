#!/bin/bash
set -e
COLLECTION="${1:-documents}"
TOP_K="${2:-5}"

cd "$(dirname "$0")/.."
echo "Starting question and answer loop..."
.venv/bin/python src/query.py --collection "$COLLECTION" --top-k "$TOP_K" --show-chunks
