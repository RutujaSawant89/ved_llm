"""TrustLM RAG — Embedding Manager.

Produces semantic embeddings for document chunks and user queries using
the sentence-transformers library with the all-MiniLM-L6-v2 model.

CRITICAL DISTINCTION — Two kinds of embeddings in TrustLM:
─────────────────────────────────────────────────────────
1. TOKEN EMBEDDINGS  (model/embeddings.py)
   - Used INSIDE the Transformer architecture.
   - Map discrete token IDs → dense vectors (shape: vocab_size × d_model).
   - Learned during language-model pretraining.
   - Purpose: feed tokens into the attention layers for next-token prediction.

2. RETRIEVAL EMBEDDINGS  (this file)
   - Used for SEMANTIC SEARCH over the knowledge base.
   - Produced by all-MiniLM-L6-v2 (a separate, pre-trained bi-encoder).
   - Shape: 384-dimensional float vectors.
   - Purpose: measure semantic similarity between query and stored chunks
     so that ChromaDB can retrieve the most relevant passages.

These two systems are completely independent. Do NOT confuse them.
─────────────────────────────────────────────────────────

Model: sentence-transformers/all-MiniLM-L6-v2
  - 22M parameters, produces 384-dim dense vectors.
  - Optimized for semantic similarity and retrieval tasks.
  - Trained on 1B+ sentence pairs.

Design decisions:
  - Model is loaded ONCE at construction time and reused for all calls.
    (Loading SentenceTransformer on every call is expensive — ~200ms.)
  - Batch embedding is supported to avoid serial per-chunk overhead.
  - Embeddings are returned as plain Python lists (ChromaDB-compatible).
"""

from __future__ import annotations

import logging
from typing import List, Optional

# Embedding dimension of all-MiniLM-L6-v2
EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages semantic retrieval embeddings for the RAG pipeline.

    These embeddings are used for semantic document retrieval and are
    SEPARATE from TrustLM's internal token embeddings.

    Args:
        model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2).
        device:     Device string ('cpu', 'cuda', etc.).  None = auto-select.
        batch_size: Number of texts to encode in one forward pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # Lazy-loaded on first use
        self._device = device
        logger.info("[EMBED] EmbeddingManager initialized (model=%s).", model_name)

    # ------------------------------------------------------------------ #
    # Lazy model loading                                                   #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        """Load the SentenceTransformer model on first call (lazy initialization).

        Why lazy?  The model takes ~200ms to load.  We defer this until the
        first actual embedding request so that simply importing the RAG package
        does not incur the startup cost.
        """
        if self._model is None:
            try:
                # Prevent sentence_transformers from triggering TensorFlow import
                # (TF has a protobuf version conflict in this environment).
                import os as _os
                _os.environ.setdefault("USE_TF", "0")
                _os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "[EMBED] sentence-transformers is not installed. "
                    "Install it with:  pip install sentence-transformers"
                ) from exc

            logger.info("[EMBED] Loading embedding model '%s'…", self.model_name)
            kwargs = {}
            if self._device:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            logger.info(
                "[EMBED] Model loaded. Embedding dimension: %d.",
                self.embedding_dim,
            )
        return self._model

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embedding vectors."""
        return EMBEDDING_DIM

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: Input text.

        Returns:
            List of floats with length == embedding_dim (384).

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("[EMBED] Cannot embed empty text.")
        result = self._get_model().encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return result[0].tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query (alias for embed_text with a descriptive name).

        Args:
            query: User's question string.

        Returns:
            List of 384 floats.
        """
        if not query or not query.strip():
            raise ValueError("[EMBED] Query cannot be empty.")
        logger.debug("[EMBED] Embedding query: '%s…'", query[:60])
        return self.embed_text(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunk texts.

        Batching is more efficient than calling embed_text() in a loop because
        the SentenceTransformer can parallelise computation across the batch.

        Args:
            texts: List of non-empty text strings.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If texts list is empty or any text is empty.
        """
        if not texts:
            raise ValueError("[EMBED] texts list is empty.")

        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty_indices:
            raise ValueError(
                f"[EMBED] Empty text at indices: {empty_indices}. "
                "All texts must be non-empty before embedding."
            )

        logger.info("[EMBED] Generating embeddings for %d texts…", len(texts))
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info("[EMBED] Generated %d embedding vectors (dim=%d).", len(vectors), EMBEDDING_DIM)
        return [v.tolist() for v in vectors]
