"""
test_rag.py — Testes unitários dos componentes principais do sistema RAG.
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
        chunks = split_text("Este é um parágrafo de teste com conteúdo suficiente para ser incluído.")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)


class TestSplitTextByParagraphs(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        text = "Parágrafo um com conteúdo suficiente.\n\nParágrafo dois com conteúdo suficiente."
        chunks = split_text_by_paragraphs(text)
        self.assertEqual(len(chunks), 2)

    def test_discards_short_paragraphs(self):
        text = "ok\n\n" + ("palavra " * 10)
        chunks = split_text_by_paragraphs(text, min_length=30)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 30)

    def test_preserves_long_paragraphs(self):
        long_para = "palavra " * 100  # ~700 chars, would be split by split_text
        text = long_para + "\n\n" + "Segundo parágrafo com texto suficiente para ser incluído."
        chunks = split_text_by_paragraphs(text)
        # O primeiro parágrafo deve ser preservado inteiro
        self.assertTrue(any(len(c) > 400 for c in chunks))


class TestSplitTextSemantic(unittest.TestCase):
    """Testes para a estratégia de chunking semântico.

    As chamadas de embedding são mockadas para garantir execução rápida
    e sem dependência de rede ou modelo carregado.
    """

    def _make_mock_model(self):
        """Retorna um mock de SentenceTransformer com encode determinístico."""
        mock = MagicMock()

        def fake_encode(texts, convert_to_numpy=True, show_progress_bar=False, **kwargs):
            n = len(texts)
            # Vetores que alternam entre dois clusters opostos, garantindo
            # que o SemanticChunker detecte variação de distância.
            vectors = np.zeros((n, 384), dtype=np.float32)
            for i in range(n):
                vectors[i, i % 2] = 1.0
                # Pequeno ruído para evitar distâncias idênticas
                rng = np.random.default_rng(i)
                vectors[i] += rng.standard_normal(384).astype(np.float32) * 0.01
            return vectors

        mock.encode.side_effect = fake_encode
        return mock

    def test_returns_list(self):
        """Verifica que split_text_semantic retorna uma lista não-vazia."""
        text = (
            "Cães são animais domésticos muito comuns. "
            "Eles são leais e amigáveis com seus donos. "
            "Cães existem como companheiros humanos há milênios.\n\n"
            "A física quântica estuda o comportamento de partículas subatômicas. "
            "O princípio da incerteza de Heisenberg é um conceito fundamental. "
            "Os fótons são partículas de luz sem massa de repouso."
        )
        chunks = split_text_semantic(text, self._make_mock_model())
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_discards_short_chunks(self):
        """Chunks abaixo de min_length devem ser descartados."""
        text = ". ".join(
            ["Sentença de teste com conteúdo suficiente número " + str(i) for i in range(10)]
        ) + "."
        chunks = split_text_semantic(text, self._make_mock_model(), min_length=50)
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 50)

    def test_accepts_all_threshold_types(self):
        """Deve aceitar os três tipos de threshold sem levantar exceções."""
        text = (
            "Primeira sentença sobre cães. Segunda sentença sobre cães. "
            "Terceira sentença sobre gatos. Quarta sentença sobre gatos."
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
        chunks = ["Texto de teste para embedding."]
        vectors = generate_embeddings(chunks, self.model)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)

    def test_multiple_chunks(self):
        chunks = ["Primeiro trecho.", "Segundo trecho.", "Terceiro trecho."]
        vectors = generate_embeddings(chunks, self.model)
        self.assertEqual(len(vectors), 3)


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_question(self):
        chunks = [{"text": "Contexto relevante.", "score": 0.9}]
        prompt = build_prompt("Qual é a resposta?", chunks)
        self.assertIn("Qual é a resposta?", prompt)

    def test_prompt_contains_context(self):
        chunks = [{"text": "Contexto relevante.", "score": 0.9}]
        prompt = build_prompt("Pergunta?", chunks)
        self.assertIn("Contexto relevante.", prompt)

    def test_prompt_format(self):
        chunks = [{"text": "Trecho.", "score": 0.8}]
        prompt = build_prompt("Pergunta?", chunks)
        self.assertIn("Contexto:", prompt)
        self.assertIn("Pergunta:", prompt)
        self.assertIn("Resposta:", prompt)


if __name__ == "__main__":
    unittest.main()
