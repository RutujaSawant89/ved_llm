"""Tests for ChromaDB Vector Store.

Tests:
  - Adding chunks
  - Similarity search
  - Top-k retrieval
  - Persistence (in-process test of upsert/query cycle)
  - Document deletion
  - Empty collection behavior
  - Metadata preservation
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def _make_store(tmp_dir: Path, collection: str = "test_coll"):
    """Helper: create a VectorStore using a temporary ChromaDB path."""
    from rag.vector_store import VectorStore
    return VectorStore(db_path=str(tmp_dir / "chroma"), collection_name=collection)


def _random_embedding(seed: int = 0, dim: int = 384):
    """Return a deterministic pseudo-random unit vector."""
    import math
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class TestVectorStore(unittest.TestCase):
    """Unit tests for VectorStore (ChromaDB backend)."""

    def setUp(self):
        """Create a fresh temporary vector store for each test."""
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.store = _make_store(self.tmp_dir)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── Add and count ─────────────────────────────────────────────── #

    def test_add_chunks_basic(self):
        """Adding chunks increases the collection count."""
        emb = _random_embedding(seed=1)
        self.store.add_chunks(
            chunk_ids=["chunk_001"],
            embeddings=[emb],
            texts=["Hello, this is a test chunk."],
            metadatas=[{"document_id": "doc1", "document_name": "doc1.txt",
                        "source": "/doc1.txt", "chunk_index": 0}],
        )
        self.assertEqual(self.store.count(), 1)

    def test_add_multiple_chunks(self):
        """Adding N chunks results in count() == N."""
        for i in range(5):
            self.store.add_chunks(
                chunk_ids=[f"chunk_{i:03d}"],
                embeddings=[_random_embedding(seed=i)],
                texts=[f"Chunk {i} text content here."],
                metadatas=[{"document_id": "docA", "document_name": "docA.txt",
                            "source": "/docA.txt", "chunk_index": i}],
            )
        self.assertEqual(self.store.count(), 5)

    def test_upsert_same_id(self):
        """Upserting a chunk with the same ID updates rather than duplicates."""
        emb = _random_embedding(seed=42)
        self.store.add_chunks(
            chunk_ids=["stable_id"],
            embeddings=[emb],
            texts=["Original text."],
            metadatas=[{"document_id": "doc1", "document_name": "doc1.txt",
                        "source": "/doc1.txt", "chunk_index": 0}],
        )
        self.store.add_chunks(
            chunk_ids=["stable_id"],
            embeddings=[emb],
            texts=["Updated text."],
            metadatas=[{"document_id": "doc1", "document_name": "doc1.txt",
                        "source": "/doc1.txt", "chunk_index": 0}],
        )
        # Count should still be 1, not 2
        self.assertEqual(self.store.count(), 1)

    def test_add_empty_raises(self):
        """Adding empty lists raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.add_chunks(
                chunk_ids=[], embeddings=[], texts=[], metadatas=[]
            )

    def test_add_mismatched_lengths_raises(self):
        """Mismatched list lengths raise ValueError."""
        with self.assertRaises(ValueError):
            self.store.add_chunks(
                chunk_ids=["a", "b"],
                embeddings=[_random_embedding()],
                texts=["text1", "text2"],
                metadatas=[{}, {}],
            )

    # ── Similarity search ─────────────────────────────────────────── #

    def test_similarity_search_returns_results(self):
        """similarity_search returns results after adding chunks."""
        emb = _random_embedding(seed=10)
        self.store.add_chunks(
            chunk_ids=["s_001"],
            embeddings=[emb],
            texts=["Machine learning is fascinating."],
            metadatas=[{"document_id": "d1", "document_name": "d1.txt",
                        "source": "/d1.txt", "chunk_index": 0}],
        )
        results = self.store.similarity_search(query_embedding=emb, top_k=1)
        self.assertEqual(len(results["ids"][0]), 1)
        self.assertEqual(results["ids"][0][0], "s_001")

    def test_similarity_search_top_k(self):
        """top_k parameter limits the number of returned results."""
        for i in range(10):
            self.store.add_chunks(
                chunk_ids=[f"tk_{i}"],
                embeddings=[_random_embedding(seed=i + 100)],
                texts=[f"Document chunk {i}."],
                metadatas=[{"document_id": f"d{i}", "document_name": f"d{i}.txt",
                            "source": f"/d{i}.txt", "chunk_index": i}],
            )

        results = self.store.similarity_search(
            query_embedding=_random_embedding(seed=999), top_k=3
        )
        self.assertEqual(len(results["ids"][0]), 3)

    def test_similarity_search_empty_collection(self):
        """similarity_search returns empty lists when collection is empty."""
        results = self.store.similarity_search(
            query_embedding=_random_embedding(seed=1), top_k=5
        )
        self.assertEqual(len(results["ids"][0]), 0)

    def test_similarity_search_empty_embedding_raises(self):
        """similarity_search raises ValueError for empty embedding."""
        with self.assertRaises(ValueError):
            self.store.similarity_search(query_embedding=[], top_k=3)

    def test_similarity_distances_are_returned(self):
        """Similarity search results include distance scores."""
        emb = _random_embedding(seed=77)
        self.store.add_chunks(
            chunk_ids=["dist_test"],
            embeddings=[emb],
            texts=["Distance test document."],
            metadatas=[{"document_id": "dd", "document_name": "dd.txt",
                        "source": "/dd.txt", "chunk_index": 0}],
        )
        results = self.store.similarity_search(query_embedding=emb, top_k=1)
        distances = results["distances"][0]
        self.assertEqual(len(distances), 1)
        self.assertIsInstance(distances[0], float)

    # ── Metadata preservation ─────────────────────────────────────── #

    def test_metadata_preserved_after_search(self):
        """Metadata stored at ingest time is returned correctly during search."""
        emb = _random_embedding(seed=55)
        self.store.add_chunks(
            chunk_ids=["meta_chunk"],
            embeddings=[emb],
            texts=["Metadata preservation test."],
            metadatas=[{
                "document_id": "doc_meta_001",
                "document_name": "meta_doc.txt",
                "source": "/path/to/meta_doc.txt",
                "chunk_index": 3,
                "page_number": 2,
            }],
        )
        results = self.store.similarity_search(query_embedding=emb, top_k=1)
        meta = results["metadatas"][0][0]
        self.assertEqual(meta["document_id"], "doc_meta_001")
        self.assertEqual(meta["document_name"], "meta_doc.txt")
        self.assertEqual(meta["chunk_index"], 3)

    # ── Deletion ──────────────────────────────────────────────────── #

    def test_delete_document_removes_chunks(self):
        """delete_document removes all chunks associated with a document_id."""
        doc_id = "doc_to_delete"
        for i in range(3):
            self.store.add_chunks(
                chunk_ids=[f"del_{i}"],
                embeddings=[_random_embedding(seed=i + 200)],
                texts=[f"Delete-me chunk {i}."],
                metadatas=[{"document_id": doc_id, "document_name": "del.txt",
                            "source": "/del.txt", "chunk_index": i}],
            )
        # Also add a chunk for a different document that should NOT be deleted
        self.store.add_chunks(
            chunk_ids=["keep_me"],
            embeddings=[_random_embedding(seed=300)],
            texts=["Keep this chunk."],
            metadatas=[{"document_id": "other_doc", "document_name": "other.txt",
                        "source": "/other.txt", "chunk_index": 0}],
        )

        self.assertEqual(self.store.count(), 4)
        n_deleted = self.store.delete_document(doc_id)
        self.assertEqual(n_deleted, 3)
        self.assertEqual(self.store.count(), 1)  # only "keep_me" remains

    def test_delete_nonexistent_document_returns_zero(self):
        """delete_document returns 0 if no chunks match the document_id."""
        n = self.store.delete_document("i_do_not_exist")
        self.assertEqual(n, 0)

    def test_delete_individual_chunk(self):
        """delete_chunk removes exactly one chunk."""
        self.store.add_chunks(
            chunk_ids=["c1", "c2"],
            embeddings=[_random_embedding(seed=1), _random_embedding(seed=2)],
            texts=["Chunk one.", "Chunk two."],
            metadatas=[
                {"document_id": "d1", "document_name": "d1.txt",
                 "source": "/d1.txt", "chunk_index": 0},
                {"document_id": "d1", "document_name": "d1.txt",
                 "source": "/d1.txt", "chunk_index": 1},
            ],
        )
        self.assertEqual(self.store.count(), 2)
        self.store.delete_chunk("c1")
        self.assertEqual(self.store.count(), 1)

    # ── Persistence simulation ────────────────────────────────────── #

    def test_persistence_via_reload(self):
        """Data added to a VectorStore persists when a new instance is created
        pointing to the same db_path (simulates program restart)."""
        from rag.vector_store import VectorStore
        db_path = self.tmp_dir / "persistent_db"

        # Write data with instance 1
        store1 = VectorStore(db_path=str(db_path), collection_name="persist_test")
        store1.add_chunks(
            chunk_ids=["persist_chunk"],
            embeddings=[_random_embedding(seed=42)],
            texts=["Persistent data test."],
            metadatas=[{"document_id": "pd", "document_name": "pd.txt",
                        "source": "/pd.txt", "chunk_index": 0}],
        )
        count_after_write = store1.count()

        # Open a new instance pointing to same path (simulates restart)
        store2 = VectorStore(db_path=str(db_path), collection_name="persist_test")
        count_after_reload = store2.count()

        self.assertEqual(count_after_write, count_after_reload,
            "Chunk count should be the same after 'restart' (new VectorStore instance).")


if __name__ == "__main__":
    unittest.main()
