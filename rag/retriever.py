"""TrustLM RAG — Retriever.

Converts a user query to an embedding and searches ChromaDB for the
most relevant document chunks.

Retrieval flow:
    query string
        ↓
    EmbeddingManager.embed_query()   — 384-dim cosine vector
        ↓
    VectorStore.similarity_search()  — ChromaDB approximate nearest neighbor
        ↓
    List[RetrievedChunk]             — ranked by similarity, with metadata

The Retriever deliberately exposes retrieval scores (cosine distances) so
that they can later be used by the Verification and Trust-Score modules.
Lower distance = more similar.

Design:
  - top_k is configurable via RAGConfig.
  - Retrieved chunks carry full metadata for source tracing.
  - The module does NOT modify any retrieved text — the retriever's job
    is only to find; formatting is left to ContextBuilder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its metadata and similarity score.

    Attributes:
        text:          The chunk text content.
        document_id:   SHA-256 ID of the source document.
        document_name: Filename of the source document.
        source:        Absolute path to the source file.
        chunk_index:   Position of this chunk within the document.
        page_number:   Source page (1-based; None for flat text files).
        distance:      Cosine distance from the query (lower = more similar).
        chunk_id:      Unique string ID in ChromaDB (format: docid_chunkN).
        extra_metadata: Any additional metadata fields stored at ingest time.
    """

    text: str
    document_id: str
    document_name: str
    source: str
    chunk_index: int
    page_number: Optional[int]
    distance: float
    chunk_id: str
    extra_metadata: Dict[str, Any]

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity score in [0, 1].

        ChromaDB returns cosine distance = 1 - cosine_similarity.
        We convert back so that 1.0 = identical and 0.0 = orthogonal.
        """
        return max(0.0, 1.0 - self.distance)


class Retriever:
    """Retrieve the most relevant document chunks for a user query.

    Args:
        embedding_manager: Pre-initialized EmbeddingManager.
        vector_store:      Pre-initialized VectorStore.
        top_k:             Default number of chunks to retrieve.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query:        User query string.
            top_k:        Override the default top_k for this call.
            where_filter: Optional ChromaDB metadata filter (e.g. filter by document_id).

        Returns:
            List of RetrievedChunk objects ranked by ascending cosine distance
            (most relevant first).

        Raises:
            ValueError: If query is empty.
            RuntimeError: If the vector store is unavailable.
        """
        if not query or not query.strip():
            raise ValueError("[RETRIEVE] Query cannot be empty.")

        k = top_k if top_k is not None else self.top_k
        logger.info("[RETRIEVE] Query: '%s…' | top_k=%d", query[:60], k)

        # 1. Convert query to a retrieval embedding
        query_embedding = self.embedding_manager.embed_query(query)

        # 2. Search ChromaDB
        raw = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            where=where_filter,
        )

        # 3. Parse results into RetrievedChunk objects
        chunks = self._parse_results(raw)

        logger.info(
            "[RETRIEVE] Retrieved %d chunks (top distance=%.4f, bottom=%.4f).",
            len(chunks),
            chunks[0].distance if chunks else 0.0,
            chunks[-1].distance if chunks else 0.0,
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _parse_results(self, raw: Dict[str, Any]) -> List[RetrievedChunk]:
        """Parse raw ChromaDB query results into RetrievedChunk objects."""
        chunks: List[RetrievedChunk] = []

        ids = raw.get("ids", [[]])[0]
        texts = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, texts, metas, dists):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    text=text or "",
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("document_name", "unknown"),
                    source=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    page_number=_int_or_none(meta.get("page_number")),
                    distance=float(dist),
                    chunk_id=chunk_id,
                    extra_metadata={
                        k: v for k, v in meta.items()
                        if k not in {
                            "document_id", "document_name", "source",
                            "chunk_index", "page_number",
                        }
                    },
                )
            )

        return chunks


def _int_or_none(value: Any) -> Optional[int]:
    """Convert a value to int, returning None for empty/None values."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
