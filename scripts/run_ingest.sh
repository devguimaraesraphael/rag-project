#!/bin/bash
set -e
PDF="${1:-}"
COLLECTION="${2:-documents}"
MAX_LENGTH="${3:-400}"

if [ -z "$PDF" ]; then
  echo "Usage: ./scripts/run_ingest.sh path/to/file.pdf [collection] [max_length]"
  exit 1
fi

cd "$(dirname "$0")/.."
echo "Starting ingestion of file: $PDF"
.venv/bin/python src/ingest.py --pdf "$PDF" --collection "$COLLECTION" --max-length "$MAX_LENGTH"
