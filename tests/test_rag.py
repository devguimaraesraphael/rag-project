"""
test_rag.py — Unit tests for the main RAG system components.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import split_text, split_text_by_paragraphs, split_text_semantic, generate_embeddings
from query import build_prompt, search_similar_chunks

from sentence_transformers import SentenceTransformer


class TestSplitText(unittest.TestCase):
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
        """Returns a mock SentenceTransformer with deterministic encode."""
        mock = MagicMock()

        def fake_encode(texts, convert_to_numpy=True, show_progress_bar=False, **kwargs):
            n = len(texts)
            # Vectors that alternate between two opposite clusters, ensuring
            # that SemanticChunker detects distance variation.
            vectors = np.zeros((n, 384), dtype=np.float32)
            for i in range(n):
                vectors[i, i % 2] = 1.0
                # Small noise to avoid identical distances
                rng = np.random.default_rng(i)
                vectors[i] += rng.standard_normal(384).astype(np.float32) * 0.01
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
    @classmethod
    def setUpClass(cls):
        cls.model = SentenceTransformer("all-MiniLM-L6-v2")

    def test_embedding_dimension(self):
        chunks = ["Test text for embedding."]
        vectors = generate_embeddings(chunks, self.model)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)

    def test_multiple_chunks(self):
        chunks = ["First chunk.", "Second chunk.", "Third chunk."]
        vectors = generate_embeddings(chunks, self.model)
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
