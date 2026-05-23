# RAG System Improvements for Better Semantic Search

## Problem Identified
When asking "Who is Khalil?", the system returns chunks that simply **mention** the name rather than chunks that **describe** the character. This is a limitation of pure vector similarity search.

## Solutions (Ranked by Impact)

### 1. **Reranking with Cross-Encoder** ⭐⭐⭐⭐⭐ (HIGHEST IMPACT)
**Problem it solves**: Vector similarity doesn't understand question-answer relationships.

**Implementation**:
```python
from sentence_transformers import CrossEncoder

# Load a cross-encoder model (evaluates question + chunk pairs)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(question: str, chunks: list, top_k: int = 5):
    """Rerank retrieved chunks using a cross-encoder model."""
    # Score each (question, chunk) pair
    pairs = [(question, c['text']) for c in chunks]
    scores = reranker.predict(pairs)
    
    # Sort by reranker score (not original similarity)
    for i, chunk in enumerate(chunks):
        chunk['rerank_score'] = scores[i]
    
    reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
    return reranked[:top_k]
```

**Benefits**: 
- 30-50% improvement in relevance
- Understands question-answer semantics
- Fast (runs after retrieval on ~20-50 candidates)

---

### 2. **Query Expansion / Reformulation** ⭐⭐⭐⭐
**Problem it solves**: Questions and answers use different vocabulary.

**Implementation**:
```python
def expand_query(question: str) -> str:
    """
    Expand 'Who is X?' questions to include characteristic terms.
    """
    # Pattern: "Who is X?" → "X is a person who... X characteristics... X description..."
    import re
    match = re.search(r"who is (\w+)", question.lower())
    if match:
        name = match.group(1)
        # Generate embedding for expanded query
        expanded = f"{name} is a person character description background role identity profession {question}"
        return expanded
    return question
```

**Better approach with LLM**:
```python
def hyde_expansion(question: str, model) -> str:
    """
    HyDE (Hypothetical Document Embeddings): Generate a fake answer 
    and use it to search for real answers.
    """
    prompt = f"""Generate a hypothetical answer to: {question}
    
    Include descriptive details about the person/character.
    
    Hypothetical answer:"""
    
    # Generate fake answer (doesn't need to be accurate)
    fake_answer = model.generate(prompt)
    
    # Use the fake answer to search (it will match descriptive chunks)
    return fake_answer
```

---

### 3. **Hybrid Search (Semantic + Keyword)** ⭐⭐⭐⭐
**Problem it solves**: Pure semantic search can miss obvious keyword matches.

**Implementation**:
```python
def hybrid_search(question: str, collection: str, top_k: int = 20):
    """
    Combine semantic search with keyword filtering.
    """
    # 1. Semantic search (top 50)
    semantic_results = search_similar_chunks(question, collection, top_k=50)
    
    # 2. Extract entities from question
    import re
    entities = re.findall(r'\b[A-Z][a-z]+\b', question)  # e.g., "Khalil"
    
    # 3. Boost chunks that contain entities in descriptive context
    for chunk in semantic_results:
        text = chunk['text'].lower()
        for entity in entities:
            # Boost if entity appears near descriptive words
            descriptive_pattern = f"{entity.lower()}.*(is|was|who|character|person|description)"
            if re.search(descriptive_pattern, text, re.IGNORECASE):
                chunk['score'] *= 1.5  # Boost descriptive chunks
    
    # 4. Re-sort and return top_k
    results = sorted(semantic_results, key=lambda x: x['score'], reverse=True)
    return results[:top_k]
```

---

### 4. **Better Embedding Model** ⭐⭐⭐
**Problem it solves**: all-MiniLM-L6-v2 is fast but not the most accurate.

**Better models** (in order of quality):
- `all-mpnet-base-v2` (768 dim, 2x slower, much better)
- `e5-large-v2` (1024 dim, best for retrieval)
- `bge-large-en-v1.5` (1024 dim, SOTA performance)

**Implementation**: Just change model name:
```python
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

---

### 5. **Contextual Chunk Enrichment** ⭐⭐⭐
**Problem it solves**: Chunks lack surrounding context during retrieval.

**Implementation**:
```python
def enrich_chunks_with_context(chunks: list, metadata: dict):
    """
    Add surrounding sentences to each chunk during indexing.
    """
    enriched = []
    for i, chunk in enumerate(chunks):
        # Add previous/next chunk as context
        prev_text = chunks[i-1] if i > 0 else ""
        next_text = chunks[i+1] if i < len(chunks)-1 else ""
        
        # Store original chunk but embed with context
        embed_text = f"{prev_text} {chunk} {next_text}"
        
        enriched.append({
            "original": chunk,
            "for_embedding": embed_text
        })
    return enriched
```

---

### 6. **Smarter Chunking Strategy** ⭐⭐
**Problem it solves**: Character descriptions might be split across chunks.

**Implementation**:
```python
def character_aware_chunking(text: str, char_name: str):
    """
    Create special chunks for character descriptions.
    """
    import re
    
    # Find paragraphs that describe the character (multi-sentence)
    pattern = f"({char_name}[^.]*\\.\\s*){3,}"  # 3+ sentences mentioning name
    
    descriptive_paragraphs = re.findall(pattern, text, re.IGNORECASE)
    
    # Treat these as high-value chunks
    return descriptive_paragraphs
```

---

## Recommended Implementation Order

1. **Start with Reranking** (1-2 hours) - Biggest impact, easy to add
2. **Add Query Expansion** (30 min) - Simple preprocessing step
3. **Upgrade Embedding Model** (15 min) - Need to re-index documents
4. **Implement Hybrid Search** (2-3 hours) - More complex but powerful

---

## Expected Results After Improvements

**Before** (current):
- Returns chunks with keyword "Khalil"
- No understanding of "who is" question type
- Similarity: 0.76 (just keyword matching)

**After** (with reranking + query expansion):
- Returns chunks describing Khalil's role, relationships, characteristics
- Understands question semantics
- Rerank score: prioritizes descriptive content

---

## Quick Test

To validate improvements, test with:
- "Who is X?" → Should return descriptions
- "What did X do?" → Should return actions/events
- "What is X's relationship with Y?" → Should return relational context
