"""
app.py — Flask web interface for the RAG system.

Routes:
  GET  /                  → main page
  POST /upload            → receives PDF, ingests and saves to Qdrant
  POST /query             → receives question, returns relevant chunks + prompt
  GET  /collections       → lists available collections in Qdrant
  DELETE /collection      → removes a collection
"""

import os
import sys
import tempfile
from typing import List

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))

# embedding_config
from embedding_config import load_model

from ingest import (
    extract_text,
    split_text,
    split_text_by_paragraphs,
    split_text_semantic,
    generate_embeddings,
    save_to_qdrant,
    collection_exists,
    DEFAULT_COLLECTION,
    CHUNK_MODE_SIZE,
    CHUNK_MODE_PARAGRAPH,
    CHUNK_MODE_SEMANTIC,
    SUPPORTED_EXTENSIONS,
)
from query import search_similar_chunks, build_prompt

from qdrant_client import QdrantClient

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
UPLOAD_MAX_MB = 100

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_MB * 1024 * 1024

# Load model and client once at startup
_model = None  # embedding_config — model type abstracted
_client: QdrantClient = None
_reranker = None  # CrossEncoder type - avoiding type hint due to optional import


def get_model():
    """Lazy-load embedding model singleton."""
    global _model
    if _model is None:
        _model = load_model()  # embedding_config
    return _model


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    return _client


def get_reranker():  # Returns CrossEncoder or None - avoiding type hint due to optional import
    """Load cross-encoder reranker model (lazy loading)."""
    global _reranker
    if not RERANKER_AVAILABLE:
        return None
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/collections", methods=["GET"])
def list_collections():
    try:
        client = get_client()
        cols = [c.name for c in client.get_collections().collections]
        return jsonify({"collections": cols})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file sent."}), 400

    file = request.files["file"]
    filename = file.filename or ""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return jsonify({"error": f"Unsupported format. Use: {supported}"}), 400

    collection = request.form.get("collection", DEFAULT_COLLECTION).strip()
    if not collection:
        collection = DEFAULT_COLLECTION

    max_length = int(request.form.get("max_length", 400))

    chunk_mode = request.form.get("chunk_mode", CHUNK_MODE_SIZE)
    if chunk_mode not in (CHUNK_MODE_SIZE, CHUNK_MODE_PARAGRAPH, CHUNK_MODE_SEMANTIC):
        chunk_mode = CHUNK_MODE_SIZE

    breakpoint_threshold_type = request.form.get("breakpoint_threshold_type", "percentile")
    if breakpoint_threshold_type not in ("percentile", "standard_deviation", "interquartile"):
        breakpoint_threshold_type = "percentile"
    try:
        breakpoint_threshold_amount = float(request.form.get("breakpoint_threshold_amount", 95.0))
    except (ValueError, TypeError):
        breakpoint_threshold_amount = 95.0

    print(f"\n{'='*60}")
    print(f"[UPLOAD] Starting document ingestion")
    print(f"[UPLOAD] File: {filename} ({ext})")
    print(f"[UPLOAD] Collection: {collection}")
    print(f"[UPLOAD] Chunk mode: {chunk_mode}")
    print(f"{'='*60}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        print(f"[UPLOAD] ✓ File saved temporarily")
        print(f"[UPLOAD] → Extracting text from {ext} file...")
        
        text = extract_text(tmp_path)
        print(f"[UPLOAD] ✓ Extracted {len(text)} characters")

        print(f"[UPLOAD] → Splitting text into chunks ({chunk_mode} mode)...")
        
        if chunk_mode == CHUNK_MODE_SEMANTIC:
            model = get_model()  # needed during chunking
            print(f"[UPLOAD]   → Using semantic chunking with {breakpoint_threshold_type}={breakpoint_threshold_amount}")
            chunks = split_text_semantic(
                text,
                model,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )
        elif chunk_mode == CHUNK_MODE_PARAGRAPH:
            print(f"[UPLOAD]   → Splitting by paragraphs")
            chunks = split_text_by_paragraphs(text)
        else:
            print(f"[UPLOAD]   → Splitting by size (max_length={max_length})")
            chunks = split_text(text, max_length=max_length)

        if not chunks:
            return jsonify({"error": "No chunks extracted from document."}), 422

        print(f"[UPLOAD] ✓ Created {len(chunks)} chunks")

        client = get_client()
        already_exists = collection_exists(client, collection)
        
        if already_exists:
            print(f"[UPLOAD] → Adding to existing collection '{collection}'")
        else:
            print(f"[UPLOAD] → Creating new collection '{collection}'")

        model = get_model()  # singleton — already cached if loaded above
        print(f"[UPLOAD] → Generating embeddings for {len(chunks)} chunks...")
        vectors = generate_embeddings(chunks, model)
        print(f"[UPLOAD] ✓ Generated {len(vectors)} embedding vectors (1024 dimensions)")  # embedding_config
        
        print(f"[UPLOAD] → Saving to Qdrant...")
        save_to_qdrant(chunks, vectors, collection, client)
        print(f"[UPLOAD] ✓ Saved to collection '{collection}'")
        print(f"[UPLOAD] ✅ Ingestion completed successfully!")
        print(f"{'='*60}\n")

        return jsonify({
            "message": f"{len(chunks)} chunks ingested successfully.",
            "chunks": len(chunks),
            "collection": collection,
            "collection_existed": already_exists,
        })

    except Exception as e:
        print(f"[UPLOAD] ✗ ERROR: {e}")
        print(f"{'='*60}\n")
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    collection = (data.get("collection") or DEFAULT_COLLECTION).strip()
    top_k = int(data.get("top_k", 5))
    use_rerank = bool(data.get("use_rerank", False))

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    print(f"\n{'='*60}")
    print(f"[QUERY] Starting semantic search")
    print(f"[QUERY] Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    print(f"[QUERY] Collection: {collection}")
    print(f"[QUERY] Top-K: {top_k}")
    print(f"[QUERY] Reranking: {'enabled' if use_rerank else 'disabled'}")
    print(f"{'='*60}")

    try:
        model = get_model()
        client = get_client()
        
        print(f"[QUERY] → Generating question embedding...")
        
        # Load reranker if requested
        reranker = get_reranker() if use_rerank else None
        
        if use_rerank and reranker:
            retrieve_count = max(20, int(top_k * 1.2))
            print(f"[QUERY] ✓ Reranker loaded (will retrieve {retrieve_count} candidates and rerank to top-{top_k})")
        
        print(f"[QUERY] → Searching vector database...")
        
        chunks = search_similar_chunks(
            question, model, client, collection, 
            top_k=top_k, reranker=reranker
        )
        
        print(f"[QUERY] ✓ Found {len(chunks)} chunks")
        
        if reranker and use_rerank:
            print(f"[QUERY] ✓ Reranked results (scores updated)")
        
        print(f"[QUERY] → Building AI prompt...")
        prompt = build_prompt(question, chunks)
        print(f"[QUERY] ✓ Prompt generated ({len(prompt)} characters)")
        print(f"[QUERY] ✅ Query completed successfully!")
        print(f"{'='*60}\n")

        return jsonify({
            "question": question,
            "collection": collection,
            "chunks": chunks,
            "prompt": prompt,
            "reranked": use_rerank and reranker is not None,
        })

    except Exception as e:
        print(f"[QUERY] ✗ ERROR: {e}")
        print(f"{'='*60}\n")
        return jsonify({"error": str(e)}), 500


@app.route("/collection", methods=["DELETE"])
def delete_collection():
    data = request.get_json(force=True)
    collection = (data.get("collection") or "").strip()
    if not collection:
        return jsonify({"error": "Collection name not provided."}), 400
    try:
        client = get_client()
        client.delete_collection(collection)
        return jsonify({"message": f"Collection '{collection}' removed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
