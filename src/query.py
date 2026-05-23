"""
query.py — Receives user questions, searches relevant chunks in Qdrant and generates AI response.

Usage:
    python src/query.py [--collection name] [--top-k 5]
"""

import argparse
import sys
from typing import List, Callable, Optional, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_COLLECTION = "documents"
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVE_K = 20  # Retrieve more candidates for reranking


def rerank_chunks(
    question: str,
    chunks: List[dict],
    reranker,  # CrossEncoder type - avoiding type hint due to optional import
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    Rerank chunks using a cross-encoder model for better semantic relevance.
    Cross-encoders evaluate the (question, chunk) pair directly, providing more accurate scores.
    
    Args:
        question: User's question
        chunks: List of candidate chunks from vector search
        reranker: CrossEncoder model instance
        top_k: Number of top chunks to return after reranking
    
    Returns:
        List of reranked chunks with updated scores
    """
    if not chunks:
        return []
    
    # Create (question, chunk_text) pairs for cross-encoder
    pairs = [(question, chunk['text']) for chunk in chunks]
    
    # Get reranker scores (higher = more relevant)
    rerank_scores = reranker.predict(pairs)
    
    # Update chunks with reranker scores
    for i, chunk in enumerate(chunks):
        chunk['original_score'] = chunk['score']
        chunk['score'] = float(rerank_scores[i])
    
    # Sort by reranker score and return top_k
    reranked = sorted(chunks, key=lambda x: x['score'], reverse=True)
    return reranked[:top_k]


def search_similar_chunks(
    question: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    reranker: Any = None,  # CrossEncoder type - avoiding specific type hint due to optional import
    retrieve_k: int = DEFAULT_RETRIEVE_K,
) -> List[dict]:
    """
    Generates embedding for the question and searches for the most similar chunks in the vector store.
    Optionally reranks results using a cross-encoder for improved relevance.
    
    Args:
        question: User's question
        model: SentenceTransformer for generating embeddings
        client: QdrantClient instance
        collection_name: Name of the collection to search
        top_k: Final number of chunks to return
        reranker: Optional CrossEncoder for reranking (if None, returns vector search results)
        retrieve_k: Number of candidates to retrieve for reranking (ignored if reranker=None)
    
    Returns:
        List of dicts with 'text' and 'score' (and 'original_score' if reranked)
    """
    embedding = model.encode([question], convert_to_numpy=True)[0].tolist()

    # Retrieve more candidates if reranking is enabled
    limit = retrieve_k if reranker else top_k
    
    results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=limit,
    )

    chunks = [{"text": hit.payload["text"], "score": hit.score} for hit in results]
    
    # Apply reranking if enabled
    if reranker:
        chunks = rerank_chunks(question, chunks, reranker, top_k)
    
    return chunks


def build_prompt(question: str, chunks: List[dict]) -> str:
    """Builds structured prompt with context and question for sending to AI model."""
    context = "\n\n".join(
        [f"[Chunk {i+1} — relevance: {c['score']:.2f}]\n{c['text']}" for i, c in enumerate(chunks)]
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def default_ai_model(prompt: str) -> str:
    """
    Default AI model (placeholder).
    Replace this function with a real integration with OpenAI, Ollama, Llama, etc.
    """
    print("\n--- GENERATED PROMPT ---")
    print(prompt)
    print("-----------------------")
    return "(Integrate your AI model here: OpenAI, Ollama, Llama, etc.)"


def query_loop(
    collection_name: str,
    top_k: int,
    ai_model_fn: Callable[[str], str] = default_ai_model,
    show_chunks: bool = False,
    use_reranker: bool = False,
) -> None:
    """Interactive loop: receives questions, searches context and displays AI response."""
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    
    # Load reranker if requested
    reranker = None
    if use_reranker:
        if not RERANKER_AVAILABLE:
            print("⚠️  CrossEncoder not available. Install: pip install sentence-transformers")
            print("Continuing without reranking...\n")
        else:
            print(f"Loading reranker model ({RERANKER_MODEL})...")
            reranker = CrossEncoder(RERANKER_MODEL)
            print("✓ Reranking enabled (retrieves top-20, reranks to top-K)\n")

    mode = " (with reranking)" if reranker else ""
    print(f"\nRAG system ready{mode}. Collection: '{collection_name}' | Top-K: {top_k}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Exiting.")
            break

        try:
            chunks = search_similar_chunks(
                question, model, client, collection_name, 
                top_k=top_k, reranker=reranker
            )
        except Exception as e:
            print(f"Error searching vector database: {e}", file=sys.stderr)
            continue

        if not chunks:
            print("No relevant chunks found.\n")
            continue

        if show_chunks:
            print("\nRecovered chunks:")
            for i, c in enumerate(chunks):
                print(f"  [{i+1}] (score={c['score']:.2f}) {c['text'][:100]}...")

        prompt = build_prompt(question, chunks)
        answer = ai_model_fn(prompt)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question and answer loop with RAG.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name in Qdrant")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--show-chunks", action="store_true", help="Display retrieved chunks before answer")
    parser.add_argument(
        "--rerank", 
        action="store_true", 
        help="Enable reranking with cross-encoder (retrieves 20 candidates, reranks to top-K)"
    )
    args = parser.parse_args()

    try:
        query_loop(
            args.collection, 
            args.top_k, 
            show_chunks=args.show_chunks,
            use_reranker=args.rerank
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
