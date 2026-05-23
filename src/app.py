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

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
UPLOAD_MAX_MB = 100

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_MB * 1024 * 1024

# Load model and client once at startup
_model: SentenceTransformer = None
_client: QdrantClient = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    return _client


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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        text = extract_text(tmp_path)

        if chunk_mode == CHUNK_MODE_SEMANTIC:
            model = get_model()  # needed during chunking
            chunks = split_text_semantic(
                text,
                model,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )
        elif chunk_mode == CHUNK_MODE_PARAGRAPH:
            chunks = split_text_by_paragraphs(text)
        else:
            chunks = split_text(text, max_length=max_length)

        if not chunks:
            return jsonify({"error": "No chunks extracted from document."}), 422

        client = get_client()
        already_exists = collection_exists(client, collection)

        model = get_model()  # singleton — already cached if loaded above
        vectors = generate_embeddings(chunks, model)
        save_to_qdrant(chunks, vectors, collection, client)

        return jsonify({
            "message": f"{len(chunks)} chunks ingested successfully.",
            "chunks": len(chunks),
            "collection": collection,
            "collection_existed": already_exists,
        })

    except Exception as e:
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

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        model = get_model()
        client = get_client()
        chunks = search_similar_chunks(question, model, client, collection, top_k)
        prompt = build_prompt(question, chunks)

        return jsonify({
            "question": question,
            "collection": collection,
            "chunks": chunks,
            "prompt": prompt,
        })

    except Exception as e:
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
