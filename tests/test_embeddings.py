"""Tests for EmbeddingManager.

Tests:
  - Embedding generation for single text
  - Embedding dimensions (384 for all-MiniLM-L6-v2)
  - Batch embedding
  - Query embedding
  - Empty input rejection
  - Model loaded once (not re-initialized)
"""

from __future__ import annotations

import unittest


class TestEmbeddingManager(unittest.TestCase):
    """Unit tests for EmbeddingManager using all-MiniLM-L6-v2."""

    @classmethod
    def setUpClass(cls):
        """Load the embedding model once for all tests in this class.

        Loading the model takes ~200ms; sharing across tests is much faster
        than loading it per-test.
        """
        from rag.embeddings import EmbeddingManager
        cls.manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        # Warm up (loads the model)
        cls.manager.embed_text("warm up")

    # ── Dimension tests ──────────────────────────────────────────── #

    def test_embedding_dimension(self):
        """embed_text returns a 384-dimensional vector."""
        from rag.embeddings import EMBEDDING_DIM
        vec = self.manager.embed_text("The capital of France is Paris.")
        self.assertEqual(len(vec), EMBEDDING_DIM)
        self.assertEqual(len(vec), 384)

    def test_embedding_is_list_of_floats(self):
        """embed_text returns a list of floats."""
        vec = self.manager.embed_text("Test sentence.")
        self.assertIsInstance(vec, list)
        for v in vec[:5]:
            self.assertIsInstance(v, float)

    # ── Single text embedding ────────────────────────────────────── #

    def test_embed_text_basic(self):
        """embed_text works for a normal input."""
        vec = self.manager.embed_text("Machine learning is a subfield of AI.")
        self.assertEqual(len(vec), 384)

    def test_embed_text_empty_raises(self):
        """embed_text raises ValueError for empty input."""
        with self.assertRaises(ValueError):
            self.manager.embed_text("")

    def test_embed_text_whitespace_raises(self):
        """embed_text raises ValueError for whitespace-only input."""
        with self.assertRaises(ValueError):
            self.manager.embed_text("   ")

    # ── Query embedding ──────────────────────────────────────────── #

    def test_embed_query(self):
        """embed_query returns a 384-dim vector."""
        vec = self.manager.embed_query("What is machine learning?")
        self.assertEqual(len(vec), 384)

    def test_embed_query_empty_raises(self):
        """embed_query raises ValueError for empty query."""
        with self.assertRaises(ValueError):
            self.manager.embed_query("")

    # ── Batch embedding ──────────────────────────────────────────── #

    def test_embed_documents_batch(self):
        """embed_documents returns one vector per input text."""
        texts = [
            "First document about machine learning.",
            "Second document about deep learning.",
            "Third document about natural language processing.",
        ]
        vecs = self.manager.embed_documents(texts)
        self.assertEqual(len(vecs), 3)
        for vec in vecs:
            self.assertEqual(len(vec), 384)

    def test_embed_documents_single(self):
        """embed_documents works for a single-element list."""
        vecs = self.manager.embed_documents(["Only one document."])
        self.assertEqual(len(vecs), 1)
        self.assertEqual(len(vecs[0]), 384)

    def test_embed_documents_empty_list_raises(self):
        """embed_documents raises ValueError for empty list."""
        with self.assertRaises(ValueError):
            self.manager.embed_documents([])

    def test_embed_documents_empty_item_raises(self):
        """embed_documents raises ValueError if any item is empty."""
        with self.assertRaises(ValueError):
            self.manager.embed_documents(["Valid text", "", "Another valid text"])

    # ── Semantic correctness ─────────────────────────────────────── #

    def test_similar_texts_have_higher_similarity(self):
        """Semantically similar texts produce more similar embeddings than unrelated ones."""
        import numpy as np

        vec_a = np.array(self.manager.embed_text("Paris is the capital of France."))
        vec_b = np.array(self.manager.embed_text("France's capital city is Paris."))
        vec_c = np.array(self.manager.embed_text("The boiling point of water is 100 degrees."))

        # Cosine similarity between a and b (similar) vs a and c (different)
        sim_ab = float(np.dot(vec_a, vec_b))  # vectors are already normalized
        sim_ac = float(np.dot(vec_a, vec_c))

        self.assertGreater(sim_ab, sim_ac,
            f"Expected similar texts to have higher similarity: sim_ab={sim_ab:.3f}, sim_ac={sim_ac:.3f}")

    def test_different_queries_produce_different_embeddings(self):
        """Two different queries produce different embedding vectors."""
        import numpy as np
        vec1 = np.array(self.manager.embed_text("What is machine learning?"))
        vec2 = np.array(self.manager.embed_text("What is the capital of France?"))
        self.assertFalse(np.allclose(vec1, vec2),
            "Expected different queries to produce different embeddings.")

    # ── Model singleton ──────────────────────────────────────────── #

    def test_model_loaded_only_once(self):
        """The underlying model object is the same instance across calls."""
        model1 = self.manager._get_model()
        model2 = self.manager._get_model()
        self.assertIs(model1, model2,
            "EmbeddingManager should not reload the model on every call.")

    def test_embedding_dim_property(self):
        """embedding_dim property returns 384."""
        self.assertEqual(self.manager.embedding_dim, 384)


if __name__ == "__main__":
    unittest.main()
