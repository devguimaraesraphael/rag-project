#!/usr/bin/env python3
"""
verify_refactoring.py — Verifies that the model-agnostic refactoring is correctly implemented.

Tests:
1. embedding_config.py exposes the required interface
2. All files use embedding_config, not direct model imports
3. Model loading and encoding functions work correctly
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_embedding_config_interface():
    """Verify embedding_config exposes required interface."""
    print("\n" + "="*70)
    print("TEST 1: Verify embedding_config.py interface")
    print("="*70)
    
    try:
        from embedding_config import VECTOR_SIZE, load_model, encode_texts, encode_query
        print("✅ All required exports found: VECTOR_SIZE, load_model, encode_texts, encode_query")
        
        assert isinstance(VECTOR_SIZE, int), "VECTOR_SIZE must be int"
        print(f"✅ VECTOR_SIZE = {VECTOR_SIZE} (int)")
        
        assert callable(load_model), "load_model must be callable"
        print("✅ load_model is callable")
        
        assert callable(encode_texts), "encode_texts must be callable"
        print("✅ encode_texts is callable")
        
        assert callable(encode_query), "encode_query must be callable"
        print("✅ encode_query is callable")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_model_agnostic_imports():
    """Verify that ingest/query/app only import from embedding_config."""
    print("\n" + "="*70)
    print("TEST 2: Verify model-agnostic imports")
    print("="*70)
    
    files_to_check = [
        "src/ingest.py",
        "src/query.py",
        "src/app.py",
    ]
    
    forbidden_imports = [
        "from sentence_transformers import SentenceTransformer",
        "from FlagEmbedding import BGEM3FlagModel",
        "import sentence_transformers",
        "SentenceTransformer(",
        "BGEM3FlagModel(",
    ]
    
    all_clean = True
    
    for filepath in files_to_check:
        print(f"\n📄 Checking {filepath}...")
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            continue
        
        with open(filepath, "r") as f:
            content = f.read()
        
        found_issues = []
        for forbidden in forbidden_imports:
            if forbidden in content:
                found_issues.append(forbidden)
        
        if found_issues:
            print(f"  ❌ FOUND FORBIDDEN IMPORTS:")
            for issue in found_issues:
                print(f"     - {issue}")
            all_clean = False
        else:
            print(f"  ✅ No forbidden imports found")
        
        # Verify it imports from embedding_config
        if "from embedding_config import" in content:
            print(f"  ✅ Imports from embedding_config")
        else:
            print(f"  ⚠️  WARNING: Does not import from embedding_config")
    
    return all_clean


def test_model_loading():
    """Test that model loads and encoding functions work."""
    print("\n" + "="*70)
    print("TEST 3: Test model loading and encoding")
    print("="*70)
    
    try:
        from embedding_config import load_model, encode_texts, encode_query, VECTOR_SIZE
        
        print("📦 Loading model...")
        model = load_model()
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        print("\n🔢 Testing encode_query()...")
        test_query = "What is the meaning of life?"
        query_vec = encode_query(model, test_query)
        
        assert isinstance(query_vec, list), "encode_query must return list"
        assert len(query_vec) == VECTOR_SIZE, f"Query vector must have {VECTOR_SIZE} dimensions"
        assert all(isinstance(x, float) for x in query_vec), "Vector elements must be floats"
        
        print(f"✅ encode_query returned {len(query_vec)}-dimensional vector")
        print(f"   Sample values: [{query_vec[0]:.4f}, {query_vec[1]:.4f}, {query_vec[2]:.4f}, ...]")
        
        print("\n📚 Testing encode_texts()...")
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level programming language.",
            "Machine learning models learn patterns from data."
        ]
        text_vecs = encode_texts(model, test_texts, show_progress=False)
        
        assert isinstance(text_vecs, list), "encode_texts must return list"
        assert len(text_vecs) == len(test_texts), "Must return one vector per text"
        assert all(len(vec) == VECTOR_SIZE for vec in text_vecs), f"All vectors must be {VECTOR_SIZE}-dimensional"
        
        print(f"✅ encode_texts returned {len(text_vecs)} vectors of {VECTOR_SIZE} dimensions each")
        print(f"   Text 1 sample: [{text_vecs[0][0]:.4f}, {text_vecs[0][1]:.4f}, {text_vecs[0][2]:.4f}, ...]")
        print(f"   Text 2 sample: [{text_vecs[1][0]:.4f}, {text_vecs[1][1]:.4f}, {text_vecs[1][2]:.4f}, ...]")
        print(f"   Text 3 sample: [{text_vecs[2][0]:.4f}, {text_vecs[2][1]:.4f}, {text_vecs[2][2]:.4f}, ...]")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test that ingest and query modules work with embedding_config."""
    print("\n" + "="*70)
    print("TEST 4: Integration test (import all modules)")
    print("="*70)
    
    try:
        print("📦 Importing ingest module...")
        import ingest
        print("✅ ingest.py imports successfully")
        
        print("📦 Importing query module...")
        import query
        print("✅ query.py imports successfully")
        
        print("📦 Importing app module...")
        import app
        print("✅ app.py imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "🔬 " + "="*66 + " 🔬")
    print("🔬  MODEL-AGNOSTIC REFACTORING VERIFICATION                          🔬")
    print("🔬 " + "="*66 + " 🔬")
    
    results = {
        "Interface Check": test_embedding_config_interface(),
        "Import Cleanliness": test_model_agnostic_imports(),
        "Model Loading": test_model_loading(),
        "Integration": test_integration(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} — {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED — Refactoring is correct!")
        print("="*70)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED — Review the output above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
