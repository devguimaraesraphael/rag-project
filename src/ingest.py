"""
ingest.py — Extracts text from documents (PDF, TXT, MD), splits into chunks,
            generates embeddings and saves to Qdrant.

Usage:
    python src/ingest.py --file file.pdf [--collection name] [--max-length 400] [--chunk-mode size|paragraph]
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


# ─── Text extractors ──────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts full text from a PDF file, preserving page order."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Error opening PDF '{pdf_path}': {e}")

    pages_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    if not pages_text:
        raise ValueError(f"No text extracted from PDF '{pdf_path}'.")

    return "\n".join(pages_text)


def extract_text_from_txt(file_path: str) -> str:
    """Reads text file content (TXT or MD), trying common encodings."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode '{file_path}' with any supported encoding.")


def extract_text(file_path: str) -> str:
    """Detects file type and extracts text appropriately."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported format: '{ext}'. Use: {supported}")


# ─── LangChain adapter ─────────────────────────────────────────────────────

class SentenceTransformerEmbeddingsAdapter:
    """Adapts a SentenceTransformer to the LangChain Embeddings interface.

    Allows reusing the already loaded model in the project without loading it
    a second time or depending on HuggingFaceEmbeddings.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return vector[0].tolist()


# ─── Splitting strategies ─────────────────────────────────────────────────────

def split_text(text: str, max_length: int = DEFAULT_MAX_LENGTH, min_length: int = DEFAULT_MIN_LENGTH) -> List[str]:
    """Splits text into chunks respecting max length without cutting words."""
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
    """Splits text by paragraphs separated by blank line(s)."""
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
    """Splits text into semantic chunks using LangChain's SemanticChunker.

    Generates embeddings for each sentence and cuts where the similarity jump
    between consecutive sentences exceeds the chosen threshold. The resulting
    chunks group semantically related sentences, regardless of character count.

    Args:
        text: Full text to split.
        model: Already loaded SentenceTransformer model (reused via adapter).
        breakpoint_threshold_type: Cutting criterion — 'percentile',
            'standard_deviation' or 'interquartile'.
        breakpoint_threshold_amount: Cut sensitivity. For 'percentile',
            use 0-100 (default 95); for others, a positive multiplier.
        min_length: Discards chunks with fewer characters than this value.
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


# ─── Qdrant helpers ─────────────────────────────────────────────────────────

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Returns True if the collection already exists in Qdrant."""
    existing = [c.name for c in client.get_collections().collections]
    return collection_name in existing


def generate_embeddings(chunks: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generates embeddings for each chunk in batch with progress bar."""
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def save_to_qdrant(
    chunks: List[str],
    vectors: List[List[float]],
    collection_name: str,
    client: QdrantClient,
) -> None:
    """Creates collection if needed and saves chunks using upsert in batches."""
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
    for start in tqdm(range(0, len(points), batch_size), desc="Saving to Qdrant"):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])


# ─── Main flow ────────────────────────────────────────────────────────────────

def ingest(
    file_path: str,
    collection_name: str,
    max_length: int,
    chunk_mode: str = CHUNK_MODE_SIZE,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> None:
    """Full flow: extracts text, splits, generates embeddings and saves to Qdrant."""
    print(f"Extracting text from '{file_path}'...")
    text = extract_text(file_path)

    print(f"Splitting text into chunks (mode: {chunk_mode})...")

    if chunk_mode == CHUNK_MODE_SEMANTIC:
        # Model is needed during chunking — load it first
        print("Loading embedding model for semantic chunking...")
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

    print(f"{len(chunks)} chunks generated.")

    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    vectors = generate_embeddings(chunks, model)

    print("Connecting to Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    if collection_exists(client, collection_name):
        print(f"⚠  Collection '{collection_name}' already exists — new chunks will be added.")
    print(f"Saving to collection '{collection_name}'...")
    save_to_qdrant(chunks, vectors, collection_name, client)

    print("✔  Ingestion completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingests documents (PDF, TXT, MD) into Qdrant.")
    parser.add_argument("--file", required=True, help="Path to file (PDF, TXT or MD)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name in Qdrant")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max size for each chunk (size mode)")
    parser.add_argument(
        "--chunk-mode",
        choices=[CHUNK_MODE_SIZE, CHUNK_MODE_PARAGRAPH, CHUNK_MODE_SEMANTIC],
        default=CHUNK_MODE_SIZE,
        help="Strategy: 'size' (by size), 'paragraph' (by paragraph) or 'semantic' (semantic)",
    )
    parser.add_argument(
        "--breakpoint-threshold-type",
        choices=["percentile", "standard_deviation", "interquartile"],
        default="percentile",
        help="Cut criterion for semantic mode (default: percentile)",
    )
    parser.add_argument(
        "--breakpoint-threshold-amount",
        type=float,
        default=95.0,
        help="Threshold value for semantic mode (default: 95.0)",
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
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
