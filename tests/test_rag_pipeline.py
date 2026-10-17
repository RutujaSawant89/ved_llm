"""Tests for RAGPipeline (end-to-end) and Document Deletion.

Tests:
  - Complete query flow (retrieval + context + prompt + generation)
  - Retrieval-only mode (generate=False)
  - Document ingestion through the pipeline
  - Document deletion removes all associated chunks
  - Empty query rejection
  - RAGResult structure
  - Integration test: "Paris is the capital of France" end-to-end

INTEGRATION TEST:
  1. A document "Paris is the capital of France." is added.
  2. The query "What is the capital of France?" is issued.
  3. We verify that the correct chunk is retrieved.
  4. Sources correctly point to the input document.
  5. Generation is called (real TrustLM model, output may be incoherent
     at current training stage — we don't test answer quality).
  6. The test passes if retrieval and context building work correctly,
     regardless of generation quality.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def _build_pipeline(tmp_dir: Path, generate: bool = True):
    """Build a fully wired RAGPipeline with a temporary ChromaDB."""
    from rag.chunker import TextChunker
    from rag.context_builder import ContextBuilder
    from rag.document_loader import DocumentLoader
    from rag.embeddings import EmbeddingManager
    from rag.generator import TrustLMGenerator
    from rag.pipeline import RAGPipeline
    from rag.prompt_builder import PromptBuilder
    from rag.retriever import Retriever
    from rag.vector_store import VectorStore

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(
        db_path=str(tmp_dir / "chroma"),
        collection_name="test_pipeline",
    )
    retriever = Retriever(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        top_k=3,
    )

    return RAGPipeline(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        generator=TrustLMGenerator(max_new_tokens=20),  # small for speed
        chunker=TextChunker(chunk_size=500, chunk_overlap=50),
        document_loader=DocumentLoader(),
    )


class TestRAGPipeline(unittest.TestCase):
    """Unit and integration tests for RAGPipeline."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── Ingest + Query ────────────────────────────────────────────── #

    def test_ingest_and_query_retrieval_only(self):
        """After ingesting a document, relevant chunks are retrieved."""
        pipeline = _build_pipeline(self.tmp_dir)

        # Write a small test document
        doc_path = self.tmp_dir / "test_doc.txt"
        doc_path.write_text(
            "The Transformer architecture was introduced in 2017. "
            "It uses self-attention mechanisms instead of recurrent layers. "
            "Transformers power modern language models like GPT and BERT.",
            encoding="utf-8",
        )

        # Ingest
        info = pipeline.ingest_document(doc_path)
        self.assertGreater(info["num_chunks"], 0)
        self.assertTrue(info["document_id"])

        # Query (retrieval only — no generation to avoid slow TrustLM run)
        result = pipeline.query("What is the Transformer architecture?", generate=False)

        self.assertGreater(result.num_chunks_found, 0)
        self.assertTrue(result.retrieved_chunks)
        self.assertTrue(result.sources)

        # Verify the first source points to our document
        self.assertEqual(result.sources[0].document_name, "test_doc.txt")

    def test_ingest_returns_metadata(self):
        """ingest_document returns correct metadata dict."""
        pipeline = _build_pipeline(self.tmp_dir)
        doc_path = self.tmp_dir / "meta.txt"
        doc_path.write_text("Document content for metadata test.", encoding="utf-8")

        info = pipeline.ingest_document(doc_path)
        self.assertIn("document_id", info)
        self.assertIn("document_name", info)
        self.assertIn("num_chunks", info)
        self.assertIn("source", info)

    def test_rag_result_structure(self):
        """RAGResult has all expected fields."""
        from rag.pipeline import RAGResult
        pipeline = _build_pipeline(self.tmp_dir)

        doc_path = self.tmp_dir / "struct_test.txt"
        doc_path.write_text("Paris is the capital of France.", encoding="utf-8")
        pipeline.ingest_document(doc_path)

        result = pipeline.query("capital of France", generate=False)

        self.assertIsInstance(result, RAGResult)
        self.assertIsInstance(result.query, str)
        self.assertIsInstance(result.answer, str)
        self.assertIsInstance(result.sources, list)
        self.assertIsInstance(result.retrieved_chunks, list)
        self.assertIsInstance(result.retrieval_scores, list)
        self.assertIsInstance(result.context, str)
        self.assertIsInstance(result.prompt, str)
        self.assertIsInstance(result.num_chunks_found, int)
        self.assertIsInstance(result.generator_name, str)

    def test_query_empty_question_raises(self):
        """query() raises ValueError for empty question."""
        pipeline = _build_pipeline(self.tmp_dir)
        with self.assertRaises(ValueError):
            pipeline.query("")

    def test_query_empty_knowledge_base(self):
        """query() returns zero chunks when knowledge base is empty."""
        pipeline = _build_pipeline(self.tmp_dir)
        result = pipeline.query("Anything?", generate=False)
        self.assertEqual(result.num_chunks_found, 0)
        self.assertEqual(result.retrieved_chunks, [])

    def test_get_stats(self):
        """get_stats() returns expected keys and correct chunk count."""
        pipeline = _build_pipeline(self.tmp_dir)
        stats = pipeline.get_stats()
        self.assertIn("total_chunks", stats)
        self.assertIn("collection_name", stats)
        self.assertIn("db_path", stats)
        self.assertIn("generator", stats)

    # ── Document deletion ─────────────────────────────────────────── #

    def test_delete_document_removes_chunks(self):
        """delete_document removes all chunks for a document from retrieval index.

        IMPORTANT: This is NOT machine unlearning.
        It only removes data from the RAG knowledge base (ChromaDB).
        The TrustLM Transformer weights are NOT modified.
        """
        pipeline = _build_pipeline(self.tmp_dir)

        # Ingest doc A and doc B
        doc_a = self.tmp_dir / "doc_a.txt"
        doc_a.write_text("Doc A content. " * 10, encoding="utf-8")
        doc_b = self.tmp_dir / "doc_b.txt"
        doc_b.write_text("Doc B content. " * 10, encoding="utf-8")

        info_a = pipeline.ingest_document(doc_a)
        pipeline.ingest_document(doc_b)

        total_before = pipeline.get_stats()["total_chunks"]
        self.assertGreater(total_before, 0)

        # Delete doc A
        n_deleted = pipeline.delete_document(info_a["document_id"])
        self.assertGreater(n_deleted, 0)

        total_after = pipeline.get_stats()["total_chunks"]
        self.assertEqual(total_after, total_before - n_deleted)

    def test_delete_document_affects_retrieval(self):
        """After deletion, chunks from deleted document no longer appear in retrieval."""
        pipeline = _build_pipeline(self.tmp_dir)

        # Ingest a unique document
        doc_path = self.tmp_dir / "unique_doc.txt"
        unique_phrase = "Xylophone quantum zeppelin 88847."  # unique retrieval target
        doc_path.write_text(f"Regular content. {unique_phrase} More content.", encoding="utf-8")
        info = pipeline.ingest_document(doc_path)

        # Verify retrieval works before deletion
        result_before = pipeline.query(unique_phrase, generate=False)
        self.assertGreater(result_before.num_chunks_found, 0)

        # Delete the document
        pipeline.delete_document(info["document_id"])

        # After deletion, querying should return no results for this document
        result_after = pipeline.query(unique_phrase, generate=False)
        doc_names = [c.document_name for c in result_after.retrieved_chunks]
        self.assertNotIn("unique_doc.txt", doc_names)


class TestEndToEndIntegration(unittest.TestCase):
    """Integration test: complete RAG flow with "Paris is the capital of France."

    This is the canonical integration test described in the project specification.

    Steps verified:
      1. Document is loaded.
      2. Text is chunked.
      3. Embedding is generated (all-MiniLM-L6-v2).
      4. Chunk is stored in ChromaDB.
      5. Query embedding is generated.
      6. Relevant chunk is retrieved.
      7. Context is constructed.
      8. Generator receives context (TrustLMGenerator).
      9. Response is generated.
      10. Source is returned.

    NOTE ON GENERATION:
      Step 9 uses the real TrustLM model.  At the current training stage
      (smoke test only, ~27K-15M params, minimal training), the generated
      text will NOT be a coherent answer.  The test deliberately does NOT
      assert answer quality.  It only asserts that:
        - Retrieval works (step 6).
        - The correct document appears in sources (step 10).
      Generation quality will improve as TrustLM is further trained.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_paris_france_integration(self):
        """Full RAG pipeline integration test."""
        pipeline = _build_pipeline(self.tmp_dir)

        # ── Step 1-4: Ingest document ──────────────────────────── #
        doc_path = self.tmp_dir / "facts.txt"
        doc_path.write_text(
            "Paris is the capital of France. "
            "Paris is located in northern France along the Seine River. "
            "It is the country's largest city and serves as the political, "
            "economic, and cultural center of France.",
            encoding="utf-8",
        )
        info = pipeline.ingest_document(doc_path)
        self.assertGreater(info["num_chunks"], 0, "Step 1-4: Document should produce at least 1 chunk.")

        # ── Step 5-6: Query embedding and retrieval ────────────── #
        result = pipeline.query(
            "What is the capital of France?",
            generate=True,  # Step 8-9: also run generation
        )

        self.assertGreater(result.num_chunks_found, 0,
            "Step 5-6: At least one chunk should be retrieved.")

        # ── Step 6b: Verify retrieved content is relevant ─────── #
        retrieved_texts = [c.text for c in result.retrieved_chunks]
        combined = " ".join(retrieved_texts).lower()
        self.assertIn("paris", combined,
            "Step 6: Retrieved text should mention 'Paris'.")
        self.assertIn("france", combined,
            "Step 6: Retrieved text should mention 'France'.")

        # ── Step 7: Context is constructed ────────────────────── #
        self.assertTrue(result.context.strip(),
            "Step 7: Context should be non-empty.")
        self.assertIn("paris", result.context.lower(),
            "Step 7: Context should contain 'paris'.")

        # ── Step 8: Prompt received by generator ──────────────── #
        self.assertIn("What is the capital of France?", result.prompt,
            "Step 8: Prompt should contain the original question.")
        self.assertIn("TrustLM", result.prompt,
            "Step 8: Prompt should contain TrustLM system instruction.")

        # ── Step 9: Response generated (don't test quality) ───── #
        # We only verify that generate() was called and returned a string.
        # At current training stage, the TrustLM output will be incoherent.
        self.assertIsInstance(result.answer, str,
            "Step 9: Generator should return a string response.")
        self.assertGreater(len(result.answer), 0,
            "Step 9: Generated response should be non-empty.")

        # ── Step 10: Sources returned ──────────────────────────── #
        self.assertGreater(len(result.sources), 0,
            "Step 10: At least one source should be returned.")
        self.assertEqual(result.sources[0].document_name, "facts.txt",
            "Step 10: Source should reference our input document.")

        # Retrieval scores are available for future Trust modules
        self.assertEqual(len(result.retrieval_scores), result.num_chunks_found,
            "Retrieval scores should match number of chunks.")
        for score in result.retrieval_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
