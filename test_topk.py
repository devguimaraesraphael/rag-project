#!/usr/bin/env python3
"""
Test script to verify Top-K functionality with and without reranking.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from query import search_similar_chunks

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

def test_topk():
    """Test if top_k parameter works correctly."""
    print("=" * 60)
    print("Testing Top-K Functionality")
    print("=" * 60)
    
    # Initialize
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient("localhost", port=6333)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if RERANKER_AVAILABLE else None
    
    question = "Who is Khalil?"
    collection = "documents"
    
    # Test different top_k values without reranking
    print("\n--- Test 1: WITHOUT Reranking ---")
    for k in [3, 5, 10, 15]:
        chunks = search_similar_chunks(
            question, model, client, collection,
            top_k=k, reranker=None
        )
        status = "✓" if len(chunks) == k else "✗"
        print(f"{status} top_k={k:2d} → returned {len(chunks):2d} chunks")
        if len(chunks) != k:
            print(f"   ERROR: Expected {k} but got {len(chunks)}")
    
    # Test with reranking
    if reranker:
        print("\n--- Test 2: WITH Reranking ---")
        for k in [3, 5, 10, 15]:
            chunks = search_similar_chunks(
                question, model, client, collection,
                top_k=k, reranker=reranker
            )
            status = "✓" if len(chunks) == k else "✗"
            print(f"{status} top_k={k:2d} → returned {len(chunks):2d} chunks (from 20 candidates)")
            if len(chunks) != k:
                print(f"   ERROR: Expected {k} but got {len(chunks)}")
    else:
        print("\n--- Reranking not available (CrossEncoder not installed) ---")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_topk()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
