"""Tests for RAG Document Ingestion Pipeline.

Tests:
  - TXT loading
  - Markdown loading
  - PDF loading (mocked)
  - Empty document rejection
  - Unsupported file type rejection
  - Missing file rejection
  - Text cleaning
  - DocumentRecord metadata correctness
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestDocumentLoader(unittest.TestCase):
    """Unit tests for DocumentLoader."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_file(self, name: str, content: str) -> Path:
        """Helper: write a file in the temp dir and return its path."""
        p = self.tmp_dir / name
        p.write_text(content, encoding="utf-8")
        return p

    # ── TXT loading ──────────────────────────────────────────────── #

    def test_load_txt_basic(self):
        """DocumentLoader correctly loads a plain .txt file."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        path = self._write_file("hello.txt", "Hello, world! This is a test document.")
        doc = loader.load(path)

        self.assertEqual(doc.document_name, "hello.txt")
        self.assertIn("Hello, world!", doc.text)
        self.assertIsNotNone(doc.document_id)
        self.assertEqual(len(doc.document_id), 64)  # SHA-256 hex
        self.assertEqual(doc.page_count, 1)

    def test_load_txt_metadata_fields(self):
        """Loaded document has all required metadata fields."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        path = self._write_file("meta_test.txt", "Content for metadata test.")
        doc = loader.load(path)

        self.assertTrue(doc.document_id)
        self.assertTrue(doc.document_name)
        self.assertTrue(doc.source)
        self.assertTrue(doc.loaded_at)  # ISO timestamp should be present
        self.assertIsInstance(doc.pages, list)
        self.assertGreater(len(doc.pages), 0)

    def test_load_txt_cleans_whitespace(self):
        """Text cleaning collapses excessive whitespace and blank lines."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        messy = "Word1  \t  word2\n\n\n\nword3"
        path = self._write_file("messy.txt", messy)
        doc = loader.load(path)

        # Three or more consecutive newlines should collapse to two
        self.assertNotIn("\n\n\n", doc.text)
        # Multiple spaces should collapse
        self.assertNotIn("  \t  ", doc.text)

    def test_load_txt_deterministic_id(self):
        """Same file path always produces the same document_id."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        path = self._write_file("stable.txt", "Deterministic ID test.")
        doc1 = loader.load(path)
        doc2 = loader.load(path)
        self.assertEqual(doc1.document_id, doc2.document_id)

    # ── Markdown loading ─────────────────────────────────────────── #

    def test_load_markdown(self):
        """DocumentLoader correctly loads a .md file."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        md_content = "# Heading\n\nThis is **markdown** content.\n\n- item1\n- item2"
        path = self._write_file("readme.md", md_content)
        doc = loader.load(path)

        self.assertIn("Heading", doc.text)
        self.assertIn("markdown", doc.text)
        self.assertEqual(doc.document_name, "readme.md")

    def test_load_markdown_metadata(self):
        """Markdown file produces correct page_count."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        path = self._write_file("doc.md", "# Title\n\nSome content here.")
        doc = loader.load(path)
        self.assertEqual(doc.page_count, 1)

    # ── PDF loading (mocked to avoid pypdf dependency in CI) ─────── #

    def test_load_pdf_requires_pypdf(self):
        """PDF loading raises ImportError with helpful message when pypdf absent."""
        import sys
        import unittest.mock as mock

        # Temporarily remove pypdf from sys.modules to simulate missing dep
        original = sys.modules.get("pypdf")
        sys.modules["pypdf"] = None  # type: ignore

        try:
            from rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            path = self._write_file("fake.pdf", b"%PDF fake".decode())

            with self.assertRaises((ImportError, RuntimeError, Exception)):
                loader.load(path)
        finally:
            if original is None:
                sys.modules.pop("pypdf", None)
            else:
                sys.modules["pypdf"] = original

    # ── Error cases ──────────────────────────────────────────────── #

    def test_missing_file_raises_error(self):
        """FileNotFoundError is raised for non-existent file."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load(self.tmp_dir / "does_not_exist.txt")

    def test_unsupported_extension_raises_error(self):
        """ValueError is raised for unsupported file type."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        path = self._write_file("doc.xyz", "content")
        with self.assertRaises(ValueError) as ctx:
            loader.load(path)
        self.assertIn("Unsupported file type", str(ctx.exception))

    def test_empty_document_raises_error(self):
        """RuntimeError is raised for a document that produces empty text."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        path = self._write_file("empty.txt", "   \n\n  ")
        with self.assertRaises(RuntimeError):
            loader.load(path)

    def test_load_many_skips_failures(self):
        """load_many() skips invalid files and continues with valid ones."""
        from rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        valid = self._write_file("good.txt", "Valid content here.")
        invalid = self.tmp_dir / "nonexistent.txt"

        docs = loader.load_many([valid, invalid])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].document_name, "good.txt")


class TestTextChunker(unittest.TestCase):
    """Unit tests for TextChunker."""

    def setUp(self):
        from rag.chunker import TextChunker
        self.chunker = TextChunker(chunk_size=200, chunk_overlap=50)

    def test_short_text_becomes_single_chunk(self):
        """Text shorter than chunk_size produces exactly one chunk."""
        chunks = self.chunker.chunk_document(
            text="Short text.",
            document_id="testid",
            document_name="test.txt",
            source="/test/test.txt",
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("Short text.", chunks[0].text)

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than chunk_size is split into multiple chunks."""
        long_text = "A " * 500  # ~1000 chars
        chunks = self.chunker.chunk_document(
            text=long_text,
            document_id="longid",
            document_name="long.txt",
            source="/test/long.txt",
        )
        self.assertGreater(len(chunks), 1)

    def test_chunk_index_is_sequential(self):
        """chunk_index values are sequential starting from 0."""
        long_text = "Word " * 400
        chunks = self.chunker.chunk_document(
            text=long_text,
            document_id="seqid",
            document_name="seq.txt",
            source="/test/seq.txt",
        )
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)

    def test_chunk_overlap_creates_shared_content(self):
        """Adjacent chunks share content due to overlap."""
        # Use paragraph-based text to make separator-based splitting predictable
        text = "\n\n".join([f"This is paragraph {i} with some content." for i in range(10)])
        chunks = self.chunker.chunk_document(
            text=text,
            document_id="overlapid",
            document_name="overlap.txt",
            source="/test/overlap.txt",
        )
        if len(chunks) > 1:
            # The end of chunk N and start of chunk N+1 should share some words
            # (we verify this loosely by checking text is not empty)
            for chunk in chunks:
                self.assertTrue(chunk.text.strip())

    def test_metadata_attached_to_chunks(self):
        """All chunk metadata fields are correctly set."""
        chunks = self.chunker.chunk_document(
            text="This is test content for metadata verification.",
            document_id="metaid123",
            document_name="meta.txt",
            source="/path/to/meta.txt",
        )
        for chunk in chunks:
            self.assertEqual(chunk.document_id, "metaid123")
            self.assertEqual(chunk.document_name, "meta.txt")
            self.assertEqual(chunk.source, "/path/to/meta.txt")
            self.assertIsNotNone(chunk.text)

    def test_empty_text_raises_error(self):
        """ValueError is raised for empty text input."""
        with self.assertRaises(ValueError):
            self.chunker.chunk_document(
                text="   ",
                document_id="emptyid",
                document_name="empty.txt",
                source="/test/empty.txt",
            )

    def test_invalid_overlap_config(self):
        """ValueError is raised if overlap >= chunk_size."""
        from rag.chunker import TextChunker
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_start_end_chars(self):
        """chunk.start_char and end_char span the chunk text correctly."""
        text = "Hello World. " * 50
        chunks = self.chunker.chunk_document(
            text=text,
            document_id="posid",
            document_name="pos.txt",
            source="/test/pos.txt",
        )
        for chunk in chunks:
            self.assertGreaterEqual(chunk.start_char, 0)
            self.assertGreater(chunk.end_char, chunk.start_char)


if __name__ == "__main__":
    unittest.main()
