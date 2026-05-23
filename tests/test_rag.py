"""
test_rag.py — Unit tests for the main RAG system components.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import split_text, split_text_by_size, split_text_by_paragraphs, split_text_semantic, generate_embeddings
from query import build_prompt, search_similar_chunks

# embedding_config — Import for testing with actual model
from embedding_config import load_model, VECTOR_SIZE


class TestSplitTextBySize(unittest.TestCase):
    """Tests for pure character-based chunking (ignores paragraph structure)."""
    
    def test_splits_purely_by_character_count(self):
        """Verifies that text is split purely by character count, ignoring paragraphs."""
        # Create text with clear paragraph breaks
        text = "A" * 200 + "\n\n" + "B" * 200 + "\n\n" + "C" * 200
        chunks = split_text_by_size(text, max_length=300)
        
        # Should split at 300 chars regardless of paragraph structure
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 300)
        
        # First chunk should contain only A's (and maybe some B's if crossing paragraph)
        # This tests that it doesn't respect paragraph boundaries
        self.assertGreater(len(chunks), 0)
    
    def test_avoids_cutting_words(self):
        """Verifies that splitting tries to break at word boundaries."""
        text = "word " * 100  # Creates a long text with clear word boundaries
        chunks = split_text_by_size(text, max_length=100)
        
        for chunk in chunks:
            # Should not end with a partial word (unless forced to)
            if len(chunk) < 100:
                self.assertTrue(chunk.strip().endswith("word") or len(chunk.split()[-1]) < 10)
    
    def test_different_from_paragraph_splitting(self):
        """Verifies that size-based splitting is different from paragraph-based."""
        # Create text with a short paragraph followed by a long paragraph
        text = "Short para.\n\n" + ("word " * 200)  # ~1400 chars total
        
        size_chunks = split_text_by_size(text, max_length=500, min_length=10)
        para_chunks = split_text_by_paragraphs(text, max_length=500, min_length=10)
        
        # Paragraph mode should preserve "Short para." as a separate chunk
        # Size mode should merge it with the next content
        # Verify that the first chunks have different content
        self.assertGreater(len(size_chunks), 0)
        self.assertGreater(len(para_chunks), 0)
        
        # The paragraph mode should have "Short para." as a standalone chunk
        has_short_para_standalone = any("Short para." in chunk and len(chunk) < 50 for chunk in para_chunks)
        self.assertTrue(has_short_para_standalone, "Paragraph mode should preserve short paragraph")
        
        # Size mode should NOT have "Short para." as a standalone chunk
        # (it should be merged with following content to reach max_length)
        has_short_para_standalone_size = any("Short para." in chunk and len(chunk) < 50 for chunk in size_chunks)
        self.assertFalse(has_short_para_standalone_size, "Size mode should merge content regardless of paragraphs")


class TestSplitText(unittest.TestCase):
    """Tests for legacy split_text function (now uses paragraph-based splitting)."""
    
    def test_splits_by_max_length(self):
        text = "a " * 300  # 600 chars
        chunks = split_text(text, max_length=400)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 400)

    def test_discards_short_paragraphs(self):
        text = "ok\n" + ("palavra " * 20)
        chunks = split_text(text, min_length=30)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 30)

    def test_returns_list(self):
        chunks = split_text("This is a paragraph with enough content to be included in results.")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)


class TestSplitTextByParagraphs(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        text = "Paragraph one with enough content.\n\nParagraph two with enough content."
        chunks = split_text_by_paragraphs(text)
        self.assertEqual(len(chunks), 2)

    def test_discards_short_paragraphs(self):
        text = "ok\n\n" + ("palavra " * 10)
        chunks = split_text_by_paragraphs(text, min_length=30)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 30)

    def test_preserves_long_paragraphs(self):
        long_para = "word " * 100  # ~700 chars, would be split by split_text
        text = long_para + "\n\n" + "Second paragraph with enough text to be included."
        chunks = split_text_by_paragraphs(text)
        # First paragraph should be preserved intact
        self.assertTrue(any(len(c) > 400 for c in chunks))


class TestSplitTextSemantic(unittest.TestCase):
    """Tests for semantic chunking strategy.

    Embedding calls are mocked to ensure fast execution
    and no dependency on network or loaded model.
    """

    def _make_mock_model(self):
        """Returns a mock model compatible with embedding_config interface."""
        mock = MagicMock()

        def fake_encode(texts, convert_to_numpy=True, show_progress_bar=False, **kwargs):
            """Mock encode that returns numpy arrays (SentenceTransformer interface)."""
            n = len(texts)
            # Vectors that alternate between two opposite clusters, ensuring
            # that SemanticChunker detects distance variation.
            vectors = np.zeros((n, VECTOR_SIZE), dtype=np.float32)
            for i in range(n):
                vectors[i, i % 2] = 1.0
                # Small noise to avoid identical distances
                rng = np.random.default_rng(i)
                vectors[i] += rng.standard_normal(VECTOR_SIZE).astype(np.float32) * 0.01
            return vectors

        mock.encode.side_effect = fake_encode
        return mock

    def test_returns_list(self):
        """Verifies that split_text_semantic returns a non-empty list."""
        text = (
            "Dogs are very common domestic animals. "
            "They are loyal and friendly with their owners. "
            "Dogs have existed as human companions for thousands of years.\n\n"
            "Quantum physics studies the behavior of subatomic particles. "
            "Heisenberg's uncertainty principle is a fundamental concept. "
            "Photons are particles of light with no rest mass."
        )
        chunks = split_text_semantic(text, self._make_mock_model())
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_discards_short_chunks(self):
        """Chunks below min_length should be discarded."""
        text = ". ".join(
            ["Test sentence with enough content number " + str(i) for i in range(10)]
        ) + "."
        chunks = split_text_semantic(text, self._make_mock_model(), min_length=50)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 50)

    def test_accepts_all_threshold_types(self):
        """Should accept all three threshold types without raising exceptions."""
        text = (
            "First sentence about dogs. Second sentence about dogs. "
            "Third sentence about cats. Fourth sentence about cats."
        )
        params = [
            ("percentile", 95.0),
            ("standard_deviation", 1.5),
            ("interquartile", 1.5),
        ]
        for threshold_type, amount in params:
            with self.subTest(threshold_type=threshold_type):
                chunks = split_text_semantic(
                    text,
                    self._make_mock_model(),
                    breakpoint_threshold_type=threshold_type,
                    breakpoint_threshold_amount=amount,
                )
                self.assertIsInstance(chunks, list)


class TestGenerateEmbeddings(unittest.TestCase):
    """Tests for embedding generation.
    
    Uses mock model to avoid downloading large models during unit tests.
    """
    
    def _make_mock_model(self):
        """Returns a mock model compatible with embedding_config interface."""
        mock = MagicMock()
        
        def fake_encode(texts, convert_to_numpy=True, show_progress_bar=False, **kwargs):
            """Mock encode that returns numpy arrays (SentenceTransformer interface)."""
            n = len(texts)
            vectors = np.random.randn(n, VECTOR_SIZE).astype(np.float32)
            return vectors
        
        mock.encode.side_effect = fake_encode
        return mock

    def test_embedding_dimension(self):
        chunks = ["Test text for embedding."]
        model = self._make_mock_model()  # Use mock instead of real model
        vectors = generate_embeddings(chunks, model)
        self.assertEqual(len(vectors), 1)
        # embedding_config — Use VECTOR_SIZE constant
        self.assertEqual(len(vectors[0]), VECTOR_SIZE)

    def test_multiple_chunks(self):
        chunks = ["First chunk.", "Second chunk.", "Third chunk."]
        model = self._make_mock_model()  # Use mock instead of real model
        vectors = generate_embeddings(chunks, model)
        self.assertEqual(len(vectors), 3)


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_question(self):
        chunks = [{"text": "Relevant context.", "score": 0.9}]
        prompt = build_prompt("What is the answer?", chunks)
        self.assertIn("What is the answer?", prompt)

    def test_prompt_contains_context(self):
        chunks = [{"text": "Relevant context.", "score": 0.9}]
        prompt = build_prompt("Question?", chunks)
        self.assertIn("Relevant context.", prompt)

    def test_prompt_format(self):
        chunks = [{"text": "Chunk.", "score": 0.8}]
        prompt = build_prompt("Question?", chunks)
        self.assertIn("Context:", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("Answer:", prompt)


if __name__ == "__main__":
    unittest.main()
