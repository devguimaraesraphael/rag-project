#!/bin/bash
set -e
COLLECTION="${1:-documents}"
TOP_K="${2:-5}"

cd "$(dirname "$0")/.."
echo "Iniciando loop de perguntas e respostas..."
.venv/bin/python src/query.py --collection "$COLLECTION" --top-k "$TOP_K" --show-chunks
