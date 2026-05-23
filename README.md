# RAG Project — Semantic Embedding System with Vector Database and AI Integration

A complete RAG (Retrieval-Augmented Generation) system that ingests PDFs/books, generates semantic embeddings, stores in Qdrant, and answers questions with context retrieved by similarity.

---

## Structure

```
rag-project/
├── src/
│   ├── app.py           # Flask web interface
│   ├── ingest.py        # Extracts PDF, splits into chunks, generates embeddings and saves to Qdrant
│   ├── query.py         # Searches context in Qdrant and generates prompt for AI
│   └── templates/
│       └── index.html   # Frontend of the web interface
├── scripts/
│   ├── start.sh                 # Complete initialization (venv + Qdrant + tests + web)
│   ├── stop.sh                  # Stops web server and Qdrant
│   ├── run_web.sh               # Starts only the Flask web interface
│   ├── install_dependencies.sh  # Installs Python dependencies
│   ├── start_qdrant.sh          # Starts Qdrant via Docker
│   ├── run_ingest.sh            # Executes ingestion of a PDF
│   └── run_query.sh             # Starts the question and answer loop
├── tests/
│   └── test_rag.py      # Unit tests of the main components
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.8+
- Docker (to run Qdrant)

---

## Quick Start

```bash
# Initializes everything: venv, dependencies, Qdrant, tests, and web interface
bash scripts/start.sh
```

Access the interface at **http://localhost:5000** and the Qdrant dashboard at **http://localhost:6333/dashboard**.

To stop:

```bash
bash scripts/stop.sh
```

---

## Manual Installation

```bash
# 1. Install Python dependencies
bash scripts/install_dependencies.sh

# 2. Start the Qdrant vector database
bash scripts/start_qdrant.sh
```

---

## Usage

### Web Interface

```bash
# Via complete initialization script (recommended)
bash scripts/start.sh

# Or just the web interface (Qdrant must already be running)
bash scripts/run_web.sh
# With custom port:
bash scripts/run_web.sh 8080
```

### Ingest a PDF

```bash
bash scripts/run_ingest.sh path/to/book.pdf
# Or with custom parameters:
bash scripts/run_ingest.sh book.pdf my_collection 500
```

### Ask Questions (CLI)

```bash
bash scripts/run_query.sh
# Or with custom parameters:
bash scripts/run_query.sh my_collection 10
# Or with reranking enabled (better semantic relevance):
bash scripts/run_query.sh my_collection 5 true
```

**Reranking**: Enabling reranking uses a cross-encoder model to re-score retrieved chunks, significantly improving relevance for complex questions. The system now **dynamically retrieves** candidates based on your Top-K value:

- **Top-K ≤ 17**: Retrieves 20 candidates for optimal reranking
- **Top-K > 17**: Retrieves 120% of Top-K (e.g., Top-K=100 → retrieves 120 candidates)

This ensures you always get the requested number of results with high quality reranking. See [TOPK_BUG_FIX.md](TOPK_BUG_FIX.md) for details.

---

## Progress Logging

The system provides **real-time visual feedback** during operations:

### Web Interface

- **Progress Modal**: Animated overlay shows detailed step-by-step logs
- **Timestamps**: Each operation logged with precise timing
- **Color-Coded**: Info (blue), Processing (yellow), Success (green), Error (red)
- **Auto-Dismiss**: Modal automatically closes after completion

**Upload Process Logs:**

```
📄 File: example.pdf (2.5 MB)
📦 Collection: documents
⚙️ Chunk mode: semantic
🔄 Uploading file...
✓ File uploaded
✓ Created 45 chunks
✓ Generated embeddings (384 dimensions)
⏱️ Completed in 3.2s
```

**Search Process Logs:**

```
❓ Question: Who is Khalil?
🔢 Top-K: 5
🎯 Reranking: Enabled
✓ Retrieved 20 initial results
✓ Reranked to top-5 most relevant chunks
⏱️ Completed in 0.54s
```

### Backend Console

Detailed logs appear in the Flask terminal:

```
[UPLOAD] Starting document ingestion
[UPLOAD] ✓ Extracted 15234 characters
[UPLOAD] ✓ Created 45 chunks
[UPLOAD] ✓ Generated 45 embedding vectors
[UPLOAD] ✅ Ingestion completed successfully!
```

See [PROGRESS_LOGGING.md](PROGRESS_LOGGING.md) for complete documentation.

---

## Integrate with an AI Model

In `src/query.py`, replace the `default_ai_model` function with integration to your model of choice:

```python
# Example with OpenAI
import openai

def openai_model(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

Pass the function when calling `query_loop`:

```python
query_loop(collection_name, top_k, ai_model_fn=openai_model)
```

---

## Tests

```bash
python -m pytest tests/test_rag.py -v
```

---

## Quality Checklist

- [ ] Qdrant is running (`docker ps`)
- [ ] Dependencies installed (`pip list`)
- [ ] PDF ingested without errors
- [ ] Embeddings with 384 dimensions
- [ ] Search returns relevant chunks
- [ ] Prompt contains context and question
- [ ] AI model configured and responding
- [ ] All tests passing
