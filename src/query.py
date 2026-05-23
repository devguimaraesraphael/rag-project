"""
query.py — Receives user questions, searches relevant chunks in Qdrant and generates AI response.

Usage:
    python src/query.py [--collection name] [--top-k 5]
"""

import argparse
import sys
from typing import List, Callable, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "documents"
DEFAULT_TOP_K = 5


def search_similar_chunks(
    question: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    Generates embedding for the question and searches for the most similar chunks in the vector store.
    Returns list of dicts with 'text' and 'score'.
    """
    embedding = model.encode([question], convert_to_numpy=True)[0].tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k,
    )

    return [{"text": hit.payload["text"], "score": hit.score} for hit in results]


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
) -> None:
    """Interactive loop: receives questions, searches context and displays AI response."""
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    print(f"\nRAG system ready. Collection: '{collection_name}' | Top-K: {top_k}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Exiting.")
            break

        try:
            chunks = search_similar_chunks(question, model, client, collection_name, top_k)
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
    args = parser.parse_args()

    try:
        query_loop(args.collection, args.top_k, show_chunks=args.show_chunks)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
