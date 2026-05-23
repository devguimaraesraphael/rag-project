"""
ingest.py — Extrai texto de documentos (PDF, TXT, MD), divide em trechos,
            gera embeddings e salva no Qdrant.

Uso:
    python src/ingest.py --file arquivo.pdf [--collection nome] [--max-length 400] [--chunk-mode size|paragraph]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
DEFAULT_MAX_LENGTH = 400
DEFAULT_MIN_LENGTH = 30
DEFAULT_COLLECTION = "documents"

CHUNK_MODE_SIZE = "size"
CHUNK_MODE_PARAGRAPH = "paragraph"
CHUNK_MODE_SEMANTIC = "semantic"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ─── Extratores de texto ──────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai o texto completo de um arquivo PDF, preservando a ordem das páginas."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao abrir o PDF '{pdf_path}': {e}")

    pages_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    if not pages_text:
        raise ValueError(f"Nenhum texto extraído do PDF '{pdf_path}'.")

    return "\n".join(pages_text)


def extract_text_from_txt(file_path: str) -> str:
    """Lê o conteúdo de um arquivo de texto (TXT ou MD), tentando encodings comuns."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Não foi possível decodificar '{file_path}' em nenhum encoding suportado.")


def extract_text(file_path: str) -> str:
    """Detecta o tipo do arquivo e extrai o texto adequadamente."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Formato não suportado: '{ext}'. Use: {supported}")


# ─── Adapter LangChain ──────────────────────────────────────────────────────

class SentenceTransformerEmbeddingsAdapter:
    """Adapta um SentenceTransformer para a interface Embeddings do LangChain.

    Permite reutilizar o modelo já carregado no projeto sem carregá-lo uma
    segunda vez ou depender de HuggingFaceEmbeddings.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return vector[0].tolist()


# ─── Estratégias de divisão ───────────────────────────────────────────────────

def split_text(text: str, max_length: int = DEFAULT_MAX_LENGTH, min_length: int = DEFAULT_MIN_LENGTH) -> List[str]:
    """Divide o texto em trechos respeitando o tamanho máximo sem cortar palavras."""
    chunks: List[str] = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if len(paragraph) < min_length:
            continue

        while len(paragraph) > max_length:
            split_point = paragraph.rfind(" ", 0, max_length)
            if split_point == -1:
                split_point = max_length
            chunks.append(paragraph[:split_point].strip())
            paragraph = paragraph[split_point:].strip()

        if len(paragraph) >= min_length:
            chunks.append(paragraph)

    return chunks


def split_text_by_paragraphs(text: str, min_length: int = DEFAULT_MIN_LENGTH) -> List[str]:
    """Divide o texto em parágrafos separados por linha(s) em branco."""
    raw = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    for para in raw:
        para = para.strip()
        if len(para) >= min_length:
            chunks.append(para)
    return chunks


def split_text_semantic(
    text: str,
    model: SentenceTransformer,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    min_length: int = DEFAULT_MIN_LENGTH,
) -> List[str]:
    """Divide o texto em chunks semânticos usando SemanticChunker do LangChain.

    Gera embeddings de cada sentença e corta onde o salto de similaridade
    entre sentenças consecutivas ultrapassa o threshold escolhido. Os chunks
    resultantes agrupam sentenças semanticamente relacionadas, independente
    do número de caracteres.

    Args:
        text: Texto completo a dividir.
        model: Modelo SentenceTransformer já carregado (reaproveitado via adapter).
        breakpoint_threshold_type: Critério de corte — 'percentile',
            'standard_deviation' ou 'interquartile'.
        breakpoint_threshold_amount: Sensibilidade do corte. Para 'percentile',
            use 0-100 (padrão 95); para os demais, um multiplicador positivo.
        min_length: Descarta chunks com menos caracteres que este valor.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    adapter = SentenceTransformerEmbeddingsAdapter(model)
    chunker = SemanticChunker(
        embeddings=adapter,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    raw = chunker.split_text(text)
    return [c.strip() for c in raw if len(c.strip()) >= min_length]


# ─── Qdrant helpers ───────────────────────────────────────────────────────────

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Retorna True se a collection já existe no Qdrant."""
    existing = [c.name for c in client.get_collections().collections]
    return collection_name in existing


def generate_embeddings(chunks: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Gera embeddings para cada trecho em lote com barra de progresso."""
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def save_to_qdrant(
    chunks: List[str],
    vectors: List[List[float]],
    collection_name: str,
    client: QdrantClient,
) -> None:
    """Cria a collection se necessário e salva os trechos usando upsert em lotes."""
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]

    batch_size = 256
    for start in tqdm(range(0, len(points), batch_size), desc="Salvando no Qdrant"):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])


# ─── Fluxo principal ──────────────────────────────────────────────────────────

def ingest(
    file_path: str,
    collection_name: str,
    max_length: int,
    chunk_mode: str = CHUNK_MODE_SIZE,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> None:
    """Fluxo completo: extrai texto, divide, gera embeddings e salva no Qdrant."""
    print(f"Extraindo texto de '{file_path}'...")
    text = extract_text(file_path)

    print(f"Dividindo texto em trechos (modo: {chunk_mode})...")

    if chunk_mode == CHUNK_MODE_SEMANTIC:
        # O modelo é necessário durante o chunking — carrega antes
        print("Carregando modelo de embeddings para chunking semântico...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        chunks = split_text_semantic(
            text,
            model,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
    elif chunk_mode == CHUNK_MODE_PARAGRAPH:
        chunks = split_text_by_paragraphs(text)
        model = None
    else:
        chunks = split_text(text, max_length=max_length)
        model = None

    print(f"{len(chunks)} trechos gerados.")

    if model is None:
        print("Carregando modelo de embeddings...")
        model = SentenceTransformer(EMBEDDING_MODEL)

    print("Gerando embeddings...")
    vectors = generate_embeddings(chunks, model)

    print("Conectando ao Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    if collection_exists(client, collection_name):
        print(f"⚠  Collection '{collection_name}' já existe — novos trechos serão adicionados.")
    print(f"Salvando na collection '{collection_name}'...")
    save_to_qdrant(chunks, vectors, collection_name, client)

    print("✔  Ingestão concluída com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingere documentos (PDF, TXT, MD) no Qdrant.")
    parser.add_argument("--file", required=True, help="Caminho para o arquivo (PDF, TXT ou MD)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Nome da collection no Qdrant")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tamanho máximo de cada trecho (modo size)")
    parser.add_argument(
        "--chunk-mode",
        choices=[CHUNK_MODE_SIZE, CHUNK_MODE_PARAGRAPH, CHUNK_MODE_SEMANTIC],
        default=CHUNK_MODE_SIZE,
        help="Estratégia: 'size' (por tamanho), 'paragraph' (por parágrafo) ou 'semantic' (semântico)",
    )
    parser.add_argument(
        "--breakpoint-threshold-type",
        choices=["percentile", "standard_deviation", "interquartile"],
        default="percentile",
        help="Critério de corte para modo semantic (padrão: percentile)",
    )
    parser.add_argument(
        "--breakpoint-threshold-amount",
        type=float,
        default=95.0,
        help="Valor do threshold para modo semantic (padrão: 95.0)",
    )
    args = parser.parse_args()

    try:
        ingest(
            args.file,
            args.collection,
            args.max_length,
            args.chunk_mode,
            args.breakpoint_threshold_type,
            args.breakpoint_threshold_amount,
        )
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
