"""TrustLM RAG — ChromaDB Vector Store.

Provides a persistent, content-addressable store for document chunk embeddings.

ChromaDB is a lightweight, open-source vector database that:
  - Runs entirely in-process (no server required).
  - Persists collections to disk between runs.
  - Supports L2, cosine, and inner-product similarity.
  - Handles metadata filtering alongside similarity search.

Design decisions:
  - We use a PERSISTENT client so the knowledge base survives restarts.
    (An ephemeral in-memory client would lose all indexed data on exit.)
  - Collection is loaded-or-created once at instantiation.
  - All stored chunks carry rich metadata (document_id, page, source, etc.)
    so that retrieval results are always traceable to their origin.
  - Deletion is document-level: deleting a document removes ALL its chunks.
    This is NOT model-level machine unlearning; it only removes data from
    the retrieval index.

ChromaDB distance metric: cosine distance (1 - cosine_similarity).
  Smaller distance → more similar.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default values (overridden by RAGConfig)
DEFAULT_COLLECTION = "trustlm_knowledge_base"
DEFAULT_DB_PATH = "data/knowledge_base/chroma_db"


class VectorStore:
    """ChromaDB-backed persistent vector store for RAG chunks.

    Args:
        db_path:         Directory where ChromaDB persists its files.
        collection_name: Name of the ChromaDB collection to use/create.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_db()

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Connect to (or create) the persistent ChromaDB collection."""
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "[STORE] chromadb is not installed. "
                "Install it with:  pip install chromadb"
            ) from exc

        self.db_path.mkdir(parents=True, exist_ok=True)
        logger.info("[STORE] Connecting to ChromaDB at: %s", self.db_path)

        self._client = chromadb.PersistentClient(path=str(self.db_path))

        # get_or_create_collection: safe to call on every startup
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        logger.info(
            "[STORE] Collection '%s' ready. Current chunk count: %d.",
            self.collection_name,
            self._collection.count(),
        )

    @property
    def collection(self):
        """Direct access to the underlying ChromaDB collection."""
        if self._collection is None:
            self._init_db()
        return self._collection

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add document chunks to the vector store.

        Args:
            chunk_ids:  Unique string IDs for each chunk (e.g. "docid_chunk0").
            embeddings: Pre-computed embedding vectors (one per chunk).
            texts:      Raw text of each chunk.
            metadatas:  Metadata dicts for each chunk (document_id, page, etc.).

        Raises:
            ValueError: If lists have different lengths or are empty.
        """
        if not chunk_ids:
            raise ValueError("[STORE] chunk_ids cannot be empty.")
        if not (len(chunk_ids) == len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError(
                "[STORE] chunk_ids, embeddings, texts, and metadatas must all "
                "have the same length."
            )

        # ChromaDB metadata values must be str, int, or float — sanitize
        safe_metas = [_sanitize_metadata(m) for m in metadatas]

        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=safe_metas,
        )
        logger.info("[STORE] Stored %d chunks in collection '%s'.", len(chunk_ids), self.collection_name)

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Embedding vector for the user's query.
            top_k:           Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            ChromaDB query result dict with keys:
              ids, documents, metadatas, distances
        """
        if not query_embedding:
            raise ValueError("[STORE] query_embedding is empty.")

        count = self.collection.count()
        if count == 0:
            logger.warning("[STORE] Collection is empty — no chunks to retrieve.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Clamp top_k to number of available chunks
        effective_k = min(top_k, count)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": effective_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        logger.info(
            "[STORE] Retrieved %d chunks for query (requested top_k=%d).",
            len(results["ids"][0]),
            top_k,
        )
        return results

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self.collection.count()

    # ------------------------------------------------------------------ #
    # Delete operations                                                    #
    # ------------------------------------------------------------------ #

    def delete_document(self, document_id: str) -> int:
        """Remove ALL chunks belonging to a document from the vector store.

        This removes the document's chunks from the RETRIEVAL INDEX only.
        It does NOT modify the TrustLM Transformer weights — it is NOT
        machine unlearning.  Use correct terminology: this is
        "document deletion from the RAG knowledge base".

        Args:
            document_id: The document's SHA-256 identifier.

        Returns:
            Number of chunks that were deleted.
        """
        # Query for all chunk IDs with this document_id in metadata
        results = self.collection.get(
            where={"document_id": {"$eq": document_id}},
            include=[],
        )
        ids_to_delete = results.get("ids", [])
        if not ids_to_delete:
            logger.warning(
                "[STORE] delete_document: no chunks found for document_id='%s'.",
                document_id,
            )
            return 0

        self.collection.delete(ids=ids_to_delete)
        logger.info(
            "[STORE] Deleted %d chunks for document_id='%s'.",
            len(ids_to_delete),
            document_id,
        )
        return len(ids_to_delete)

    def delete_chunk(self, chunk_id: str) -> None:
        """Delete a single chunk by its ID.

        Args:
            chunk_id: Unique chunk ID string.
        """
        self.collection.delete(ids=[chunk_id])
        logger.info("[STORE] Deleted chunk: '%s'.", chunk_id)

    def reset_collection(self) -> None:
        """WARNING: Delete and recreate the entire collection.

        Use only in testing.  This wipes all stored chunks and embeddings.
        """
        logger.warning("[STORE] Resetting collection '%s'. All data will be lost!", self.collection_name)
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("[STORE] Collection '%s' reset successfully.", self.collection_name)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata values to types accepted by ChromaDB (str, int, float).

    ChromaDB rejects None and complex types in metadata dicts.
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
