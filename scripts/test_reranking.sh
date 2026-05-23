#!/bin/bash
# Test script to compare results with and without reranking

set -e

COLLECTION="${1:-documents}"
QUESTION="${2:-Who is Khalil?}"

cd "$(dirname "$0")/.."

echo "================================================"
echo "Testing RAG Search Quality"
echo "Collection: $COLLECTION"
echo "Question: $QUESTION"
echo "================================================"
echo ""

if ! .venv/bin/python -c "from sentence_transformers import CrossEncoder" 2>/dev/null; then
  echo "⚠️  CrossEncoder not available. Reranking will be skipped."
  echo "Install: pip install sentence-transformers"
  echo ""
fi

echo "--- Test 1: Without Reranking ---"
echo "$QUESTION" | .venv/bin/python src/query.py --collection "$COLLECTION" --top-k 5 --show-chunks 2>&1 | head -30

echo ""
echo ""
echo "--- Test 2: With Reranking ---"
echo "$QUESTION" | .venv/bin/python src/query.py --collection "$COLLECTION" --top-k 5 --show-chunks --rerank 2>&1 | head -30

echo ""
echo ""
echo "================================================"
echo "Compare the results above:"
echo "- Without reranking: May show chunks that just mention keywords"
echo "- With reranking: Should show more descriptive/relevant chunks"
echo "================================================"
