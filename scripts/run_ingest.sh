#!/bin/bash
set -e
PDF="${1:-}"
COLLECTION="${2:-documents}"
MAX_LENGTH="${3:-400}"

if [ -z "$PDF" ]; then
  echo "Uso: ./scripts/run_ingest.sh caminho/para/arquivo.pdf [collection] [max_length]"
  exit 1
fi

cd "$(dirname "$0")/.."
echo "Iniciando ingestão do arquivo: $PDF"
.venv/bin/python src/ingest.py --pdf "$PDF" --collection "$COLLECTION" --max-length "$MAX_LENGTH"
