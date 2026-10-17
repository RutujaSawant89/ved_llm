"""Tests for Retriever, ContextBuilder, and PromptBuilder.

Tests:
  - Query embedding
  - Top-k retrieval
  - Metadata preservation in retrieved chunks
  - RetrievedChunk.similarity_score property
  - ContextBuilder formatting and source tracking
  - PromptBuilder prompt construction
  - Empty query rejection
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _random_embedding(seed: int = 0, dim: int = 384):
    """Deterministic pseudo-random unit vector for testing."""
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


class TestRetriever(unittest.TestCase):
    """Unit tests for Retriever."""

    def setUp(self):
        """Set up a retriever backed by an in-memory vector store."""
        self.tmp_dir = Path(tempfile.mkdtemp())

        # Build real embedding manager and vector store
        from rag.embeddings import EmbeddingManager
        from rag.retriever import Retriever
        from rag.vector_store import VectorStore

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(
            db_path=str(self.tmp_dir / "chroma"),
            collection_name="test_retriever",
        )
        self.retriever = Retriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
            top_k=3,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _populate(self, n: int = 5):
        """Helper: add n chunks with real embeddings."""
        texts = [
            "Paris is the capital of France.",
            "Machine learning is a subset of artificial intelligence.",
            "The boiling point of water is 100 degrees Celsius.",
            "The Transformer architecture uses attention mechanisms.",
            "Berlin is the capital of Germany.",
        ][:n]

        embs = self.embedding_manager.embed_documents(texts)

        chunk_ids = [f"chunk_{i}" for i in range(n)]
        metadatas = [
            {
                "document_id": f"docid_{i}",
                "document_name": f"doc_{i}.txt",
                "source": f"/path/doc_{i}.txt",
                "chunk_index": i,
                "page_number": i + 1,
            }
            for i in range(n)
        ]
        self.vector_store.add_chunks(
            chunk_ids=chunk_ids, embeddings=embs,
            texts=texts, metadatas=metadatas,
        )

    # ── Basic retrieval ──────────────────────────────────────────── #

    def test_retrieve_returns_results(self):
        """retrieve() returns a non-empty list when the collection has chunks."""
        self._populate()
        results = self.retriever.retrieve("What is the capital of France?")
        self.assertGreater(len(results), 0)

    def test_retrieve_top_k_limit(self):
        """retrieve() respects the top_k parameter."""
        self._populate(5)
        results = self.retriever.retrieve("AI and machine learning", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_returns_chunk_objects(self):
        """retrieve() returns RetrievedChunk objects with the expected fields."""
        from rag.retriever import RetrievedChunk
        self._populate()
        results = self.retriever.retrieve("machine learning")
        for chunk in results:
            self.assertIsInstance(chunk, RetrievedChunk)
            self.assertIsInstance(chunk.text, str)
            self.assertIsInstance(chunk.distance, float)
            self.assertTrue(chunk.document_name)
            self.assertTrue(chunk.chunk_id)

    def test_retrieve_metadata_preserved(self):
        """Retrieval preserves document_id, chunk_index, and page_number."""
        self._populate()
        results = self.retriever.retrieve("capital of France", top_k=1)
        self.assertGreater(len(results), 0)
        chunk = results[0]
        self.assertTrue(chunk.document_id)
        self.assertIsInstance(chunk.chunk_index, int)

    def test_retrieve_empty_query_raises(self):
        """retrieve() raises ValueError for empty query."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("")

    def test_retrieve_whitespace_query_raises(self):
        """retrieve() raises ValueError for whitespace-only query."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

    def test_retrieve_empty_collection(self):
        """retrieve() returns empty list when collection is empty."""
        results = self.retriever.retrieve("Any query here?")
        self.assertEqual(len(results), 0)

    # ── Similarity score ─────────────────────────────────────────── #

    def test_similarity_score_range(self):
        """similarity_score is in [0, 1] for all retrieved chunks."""
        self._populate()
        results = self.retriever.retrieve("capital")
        for chunk in results:
            self.assertGreaterEqual(chunk.similarity_score, 0.0)
            self.assertLessEqual(chunk.similarity_score, 1.0)

    def test_most_relevant_chunk_first(self):
        """The most relevant chunk appears first (highest similarity / lowest distance)."""
        self._populate()
        results = self.retriever.retrieve("What is the capital of France?")
        if len(results) > 1:
            # Similarity should be non-increasing
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i].similarity_score,
                    results[i + 1].similarity_score,
                )


class TestContextBuilder(unittest.TestCase):
    """Unit tests for ContextBuilder."""

    def _make_chunk(self, text, doc_name, page, idx=0, distance=0.1):
        from rag.retriever import RetrievedChunk
        return RetrievedChunk(
            text=text,
            document_id=f"docid_{idx}",
            document_name=doc_name,
            source=f"/path/{doc_name}",
            chunk_index=idx,
            page_number=page,
            distance=distance,
            chunk_id=f"cid_{idx}",
            extra_metadata={},
        )

    def test_build_returns_non_empty_context(self):
        """build() returns a non-empty context string for valid chunks."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        chunks = [self._make_chunk("Paris is the capital of France.", "ai.txt", 1)]
        context, sources = builder.build(chunks)
        self.assertTrue(context.strip())
        self.assertEqual(len(sources), 1)

    def test_build_includes_document_name(self):
        """Context string contains the document name."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        chunks = [self._make_chunk("Test content.", "my_document.pdf", 3)]
        context, sources = builder.build(chunks)
        self.assertIn("my_document.pdf", context)

    def test_build_includes_page_number(self):
        """Context string includes the page number when available."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        chunks = [self._make_chunk("Content.", "doc.pdf", page=7)]
        context, sources = builder.build(chunks)
        self.assertIn("7", context)

    def test_build_includes_chunk_text(self):
        """Context string contains the chunk text."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        chunks = [self._make_chunk("Unique marker text XYZ123.", "doc.txt", 1)]
        context, _ = builder.build(chunks)
        self.assertIn("Unique marker text XYZ123.", context)

    def test_build_multiple_sources(self):
        """Build produces multiple source blocks for multiple chunks."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        chunks = [
            self._make_chunk("First chunk.", "doc1.txt", 1, idx=0),
            self._make_chunk("Second chunk.", "doc2.txt", 2, idx=1),
        ]
        context, sources = builder.build(chunks)
        self.assertEqual(len(sources), 2)
        self.assertIn("[Source 1]", context)
        self.assertIn("[Source 2]", context)

    def test_build_empty_chunks_returns_empty(self):
        """build() returns empty context and empty sources for empty chunk list."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        context, sources = builder.build([])
        self.assertEqual(context, "")
        self.assertEqual(sources, [])

    def test_source_info_display(self):
        """SourceInfo.display() returns a formatted string."""
        from rag.context_builder import ContextBuilder, SourceInfo
        src = SourceInfo(
            source_number=1,
            document_name="test.pdf",
            document_id="abc123",
            source_path="/test.pdf",
            page_number=4,
            similarity=0.87,
            chunk_index=2,
        )
        display = src.display()
        self.assertIn("test.pdf", display)
        self.assertIn("4", display)
        self.assertIn("0.87", display)

    def test_context_max_chars_limit(self):
        """Context is truncated when max_context_chars is reached."""
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder(max_context_chars=100)
        big_chunks = [
            self._make_chunk("A" * 200, f"doc{i}.txt", i + 1, idx=i)
            for i in range(5)
        ]
        context, sources = builder.build(big_chunks)
        self.assertLessEqual(len(context), 500)  # should be truncated before all 5


class TestPromptBuilder(unittest.TestCase):
    """Unit tests for PromptBuilder."""

    def setUp(self):
        from rag.prompt_builder import PromptBuilder
        self.builder = PromptBuilder()

    def test_build_prompt_contains_query(self):
        """Prompt contains the user query."""
        prompt = self.builder.build_prompt(
            query="What is machine learning?",
            context="Machine learning is a subset of AI.",
        )
        self.assertIn("What is machine learning?", prompt)

    def test_build_prompt_contains_context(self):
        """Prompt contains the context text."""
        prompt = self.builder.build_prompt(
            query="Test question?",
            context="Unique context marker ABC987.",
        )
        self.assertIn("Unique context marker ABC987.", prompt)

    def test_build_prompt_contains_system_instruction(self):
        """Prompt contains the system instruction about reliability."""
        prompt = self.builder.build_prompt(
            query="Test?",
            context="Some context.",
        )
        self.assertIn("TrustLM", prompt)

    def test_build_prompt_answer_prefix(self):
        """Prompt ends with the answer prefix to prime the generator."""
        prompt = self.builder.build_prompt(
            query="Question?",
            context="Context.",
        )
        self.assertTrue(prompt.strip().endswith("Answer:"))

    def test_build_prompt_empty_query_raises(self):
        """build_prompt raises ValueError for empty query."""
        with self.assertRaises(ValueError):
            self.builder.build_prompt(query="", context="Some context.")

    def test_build_prompt_empty_context_handled(self):
        """build_prompt handles empty context gracefully (no crash)."""
        prompt = self.builder.build_prompt(query="A question?", context="")
        self.assertIn("A question?", prompt)

    def test_no_context_prompt(self):
        """build_no_context_prompt includes a message about missing information."""
        prompt = self.builder.build_no_context_prompt("What is AI?")
        self.assertIn("What is AI?", prompt)
        self.assertIn("No relevant context", prompt)

    def test_custom_system_instruction(self):
        """Custom system instruction overrides the default."""
        from rag.prompt_builder import PromptBuilder
        custom = PromptBuilder(system_instruction="Custom instruction here.")
        prompt = custom.build_prompt("Question?", "Context.")
        self.assertIn("Custom instruction here.", prompt)


if __name__ == "__main__":
    unittest.main()
