"""
embedding_config.py — Single source of truth for embedding model configuration.

This is the ONLY file allowed to import embedding libraries (FlagEmbedding, sentence_transformers, etc.).
To switch models, modify only this file — all other files remain unchanged.

Current model: BAAI/bge-m3 (FlagEmbedding, 1024 dimensions)

AVAILABLE ALTERNATIVE MODELS (lighter/faster):
See commented examples below for 3 alternative models you can use.
To switch models:
1. Uncomment the corresponding lines in requirements.txt
2. Install dependencies: pip install -r requirements.txt
3. Uncomment the model configuration below
4. Comment out the current model configuration
5. Update VECTOR_SIZE accordingly
6. Re-ingest all documents (vector dimensions will change!)
"""

from typing import List

# ============================================================================
# CURRENT MODEL: BAAI/bge-m3 (Production-grade, multilingual, 1024 dimensions)
# ============================================================================
from FlagEmbedding import BGEM3FlagModel
VECTOR_SIZE = 1024

def load_model():
    """Loads BAAI/bge-m3 model (2.27GB, 88 languages, best quality)."""
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def encode_texts(model, texts: List[str], show_progress: bool = False) -> List[List[float]]:
    """BGE-M3 batch encoding."""
    output = model.encode(texts, batch_size=12)
    return output["dense_vecs"].tolist()

def encode_query(model, text: str) -> List[float]:
    """BGE-M3 single query encoding."""
    output = model.encode([text])
    return output["dense_vecs"][0].tolist()


# ============================================================================
# ALTERNATIVE MODEL 1: all-MiniLM-L6-v2 (Lightweight, fast, 384 dimensions)
# ============================================================================
# Pros: Very fast, CPU-friendly, small size (90MB), good for prototyping
# Cons: English-focused, lower quality than BGE models
# 
# Uncomment below and comment out current model to use:
# ----------------------------------------------------------------------------
# from sentence_transformers import SentenceTransformer
# VECTOR_SIZE = 384
#
# def load_model():
#     """Loads all-MiniLM-L6-v2 model (90MB, fast, English-focused)."""
#     return SentenceTransformer("all-MiniLM-L6-v2")
#
# def encode_texts(model, texts: List[str], show_progress: bool = False) -> List[List[float]]:
#     """SentenceTransformer batch encoding."""
#     vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
#     return [v.tolist() for v in vectors]
#
# def encode_query(model, text: str) -> List[float]:
#     """SentenceTransformer single query encoding."""
#     vector = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
#     return vector[0].tolist()


# ============================================================================
# ALTERNATIVE MODEL 2: BAAI/bge-small-en-v1.5 (Balanced, 384 dimensions)
# ============================================================================
# Pros: Smaller BGE model, good quality/speed balance, 133MB
# Cons: English-only, not as powerful as bge-m3
# 
# Uncomment below and comment out current model to use:
# ----------------------------------------------------------------------------
# from sentence_transformers import SentenceTransformer
# VECTOR_SIZE = 384
#
# def load_model():
#     """Loads BAAI/bge-small-en-v1.5 model (133MB, English, good balance)."""
#     return SentenceTransformer("BAAI/bge-small-en-v1.5")
#
# def encode_texts(model, texts: List[str], show_progress: bool = False) -> List[List[float]]:
#     """SentenceTransformer batch encoding."""
#     vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
#     return [v.tolist() for v in vectors]
#
# def encode_query(model, text: str) -> List[float]:
#     """SentenceTransformer single query encoding."""
#     vector = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
#     return vector[0].tolist()


# ============================================================================
# ALTERNATIVE MODEL 3: paraphrase-multilingual-MiniLM-L12-v2 (Multilingual, 384 dimensions)
# ============================================================================
# Pros: Multilingual (50+ languages), lightweight (470MB), good for diverse content
# Cons: Lower quality than BGE-M3, moderate speed
# 
# Uncomment below and comment out current model to use:
# ----------------------------------------------------------------------------
# from sentence_transformers import SentenceTransformer
# VECTOR_SIZE = 384
#
# def load_model():
#     """Loads paraphrase-multilingual-MiniLM-L12-v2 (470MB, 50+ languages)."""
#     return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#
# def encode_texts(model, texts: List[str], show_progress: bool = False) -> List[List[float]]:
#     """SentenceTransformer batch encoding."""
#     vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
#     return [v.tolist() for v in vectors]
#
# def encode_query(model, text: str) -> List[float]:
#     """SentenceTransformer single query encoding."""
#     vector = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
#     return vector[0].tolist()


# ============================================================================
# HELPER FUNCTIONS (Used by both current and alternative models)
# ============================================================================
# Note: If using alternative models above, the encode_texts and encode_query
# functions are redefined in each section. The functions below are only used
# when the current model (BGE-M3) is active.



