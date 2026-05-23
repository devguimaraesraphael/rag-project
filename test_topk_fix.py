"""
Test script to validate Top-K fix for reranking.
Verifies that when Top-K > 20, the system retrieves and returns the correct number of chunks.
"""

from src.query import search_similar_chunks
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

COLLECTION = "documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def test_topk_values():
    """Test various Top-K values with and without reranking."""
    
    print("Loading models...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    
    question = "Who is Khalil?"
    
    test_cases = [
        (5, False, 5, "Small Top-K without reranking"),
        (5, True, 5, "Small Top-K with reranking (should retrieve 20 for better reranking)"),
        (20, True, 20, "Top-K=20 with reranking (should retrieve 24 candidates)"),
        (50, True, 50, "Medium Top-K=50 with reranking (should retrieve 60 candidates)"),
        (100, True, 100, "Large Top-K=100 with reranking (should retrieve 120 candidates) - BUG FIX TEST"),
    ]
    
    print("\n" + "="*80)
    print("TOP-K FIX VALIDATION TEST")
    print("="*80 + "\n")
    
    all_passed = True
    
    for top_k, use_reranker, expected_result, description in test_cases:
        print(f"\n📝 Test: {description}")
        print(f"   Top-K: {top_k} | Reranking: {'✓' if use_reranker else '✗'}")
        
        try:
            chunks = search_similar_chunks(
                question,
                model,
                client,
                COLLECTION,
                top_k=top_k,
                reranker=reranker if use_reranker else None
            )
            
            actual_count = len(chunks)
            
            if actual_count == expected_result:
                print(f"   ✅ PASS: Returned {actual_count} chunks (expected {expected_result})")
            else:
                print(f"   ❌ FAIL: Returned {actual_count} chunks (expected {expected_result})")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Top-K bug is fixed!")
    else:
        print("❌ SOME TESTS FAILED - Review the implementation")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    try:
        passed = test_topk_values()
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)
