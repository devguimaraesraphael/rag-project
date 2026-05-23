"""
query.py — Recebe perguntas do usuário, busca trechos relevantes no Qdrant e gera resposta com IA.

Uso:
    python src/query.py [--collection nome] [--top-k 5]
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
    Gera embedding da pergunta e busca os trechos mais similares no banco vetorial.
    Retorna lista de dicts com 'text' e 'score'.
    """
    embedding = model.encode([question], convert_to_numpy=True)[0].tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k,
    )

    return [{"text": hit.payload["text"], "score": hit.score} for hit in results]


def build_prompt(question: str, chunks: List[dict]) -> str:
    """Monta o prompt estruturado com contexto e pergunta para envio ao modelo de IA."""
    context = "\n\n".join(
        [f"[Trecho {i+1} — relevância: {c['score']:.2f}]\n{c['text']}" for i, c in enumerate(chunks)]
    )
    return f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"


def default_ai_model(prompt: str) -> str:
    """
    Modelo de IA padrão (placeholder).
    Substitua esta função por uma integração real com OpenAI, Ollama, Llama, etc.
    """
    print("\n--- PROMPT GERADO ---")
    print(prompt)
    print("---------------------")
    return "(Integre aqui seu modelo de IA: OpenAI, Ollama, Llama, etc.)"


def query_loop(
    collection_name: str,
    top_k: int,
    ai_model_fn: Callable[[str], str] = default_ai_model,
    show_chunks: bool = False,
) -> None:
    """Loop interativo: recebe perguntas, busca contexto e exibe resposta da IA."""
    print("Carregando modelo de embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Conectando ao Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    print(f"\nSistema RAG pronto. Collection: '{collection_name}' | Top-K: {top_k}")
    print("Digite 'sair' para encerrar.\n")

    while True:
        question = input("Pergunta: ").strip()
        if not question:
            continue
        if question.lower() == "sair":
            print("Encerrando.")
            break

        try:
            chunks = search_similar_chunks(question, model, client, collection_name, top_k)
        except Exception as e:
            print(f"Erro ao buscar no banco vetorial: {e}", file=sys.stderr)
            continue

        if not chunks:
            print("Nenhum trecho relevante encontrado.\n")
            continue

        if show_chunks:
            print("\nTrechos recuperados:")
            for i, c in enumerate(chunks):
                print(f"  [{i+1}] (score={c['score']:.2f}) {c['text'][:100]}...")

        prompt = build_prompt(question, chunks)
        answer = ai_model_fn(prompt)
        print(f"\nResposta: {answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop de perguntas e respostas com RAG.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Nome da collection no Qdrant")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Número de trechos a recuperar")
    parser.add_argument("--show-chunks", action="store_true", help="Exibir trechos recuperados antes da resposta")
    args = parser.parse_args()

    try:
        query_loop(args.collection, args.top_k, show_chunks=args.show_chunks)
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
