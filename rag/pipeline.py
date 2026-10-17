"""TrustLM RAG — Central Pipeline.

RAGPipeline is the single entry point for the complete
Retrieval-Augmented Generation workflow.

Flow:
    user query
        ↓
    Retriever.retrieve()          — embed query, search ChromaDB, get top-k chunks
        ↓
    ContextBuilder.build()        — format chunks + source metadata
        ↓
    PromptBuilder.build_prompt()  — assemble system instruction + context + question
        ↓
    BaseGenerator.generate()      — call TrustLMGenerator (or any swappable generator)
        ↓
    RAGResult                     — answer + sources + chunks + scores

The pipeline also provides:
    - ingest_document(): load → chunk → embed → store (complete ingestion)
    - delete_document(): remove all chunks for a document from ChromaDB
    - get_stats():        collection statistics

Design:
    All components are injected at construction time so they can be
    individually replaced for testing or for future upgrades (e.g. swap
    TrustLMGenerator for a fine-tuned checkpoint).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.chunker import TextChunker
from rag.context_builder import ContextBuilder, SourceInfo
from rag.document_loader import DocumentLoader
from rag.embeddings import EmbeddingManager
from rag.generator import BaseGenerator, TrustLMGenerator
from rag.prompt_builder import PromptBuilder
from rag.retriever import RetrievedChunk, Retriever
from rag.vector_store import VectorStore
from rag.query_processor import QueryProcessor
from rag.reranker import Reranker
from verification.verifier import Verifier
from trust.scorer import TrustScorer

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                         #
# ────────────────────────────────────────────────────────────────────── #


@dataclass
class RAGResult:
    """Structured output of a RAG pipeline query.

    This structure is deliberately rich so future modules (Verification,
    Uncertainty, Trust Score) can plug in without changing the pipeline
    interface.

    Attributes:
        query:            The original user query.
        answer:           Generated answer string.
        sources:          Structured source citations for the retrieved chunks.
        retrieved_chunks: Raw list of RetrievedChunk objects.
        retrieval_scores: Cosine similarity scores (one per chunk, descending).
        context:          The formatted context string that was sent to the generator.
        prompt:           The complete prompt that was sent to the generator.
        num_chunks_found: Number of chunks retrieved from ChromaDB.
        generator_name:   Name of the generator that produced the answer.
        metadata:         Arbitrary dict for future extension (e.g. trust score).
    """

    query: str
    answer: str
    sources: List[SourceInfo]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_scores: List[float]
    context: str
    prompt: str
    num_chunks_found: int
    generator_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────── #
# RAGPipeline                                                              #
# ────────────────────────────────────────────────────────────────────── #


class RAGPipeline:
    """Orchestrates the complete RAG workflow.

    Args:
        embedding_manager: EmbeddingManager instance.
        vector_store:      VectorStore instance.
        retriever:         Retriever instance.
        context_builder:   ContextBuilder instance.
        prompt_builder:    PromptBuilder instance.
        generator:         Any BaseGenerator implementation.
        chunker:           TextChunker instance.
        document_loader:   DocumentLoader instance.
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        generator: Optional[BaseGenerator] = None,
        chunker: Optional[TextChunker] = None,
        document_loader: Optional[DocumentLoader] = None,
        query_processor: Optional[QueryProcessor] = None,
        reranker: Optional[Reranker] = None,
        verifier: Optional[Verifier] = None,
        trust_scorer: Optional[TrustScorer] = None,
    ) -> None:
        # Build default components if not injected
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or Retriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
        )
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generator = generator or TrustLMGenerator()
        self.chunker = chunker or TextChunker()
        self.document_loader = document_loader or DocumentLoader()
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker or Reranker()
        self.verifier = verifier or Verifier()
        self.trust_scorer = trust_scorer or TrustScorer()

        logger.info(
            "[PIPELINE] RAGPipeline ready. Generator: %s.", self.generator.name
        )

    # ------------------------------------------------------------------ #
    # Query (main entry point)                                             #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        generate: bool = True,
    ) -> RAGResult:
        """Run a complete RAG query and return a structured result.

        Args:
            question: User's question string.
            top_k:    Override the default number of chunks to retrieve.
            generate: If False, skip generation (retrieval-only mode useful
                      for testing retrieval quality independently).

        Returns:
            RAGResult with answer, sources, chunks, and scores.

        Raises:
            ValueError: If question is empty.
        """
        if not question or not question.strip():
            raise ValueError("[PIPELINE] question cannot be empty.")

        logger.info("=" * 60)
        logger.info("[PIPELINE] Processing query: '%s…'", question[:80])

        # 0. Query Understanding
        query_info = self.query_processor.process(question)

        # 1. Retrieve (broad)
        broad_k = max(10, (top_k or 5) * 2)
        raw_chunks = self.retriever.retrieve(question, top_k=broad_k)

        # 2. Rerank (narrow)
        final_k = top_k or 5
        chunks = self.reranker.rerank(raw_chunks, query_info, top_k=final_k)

        # 3. Build context
        context, sources = self.context_builder.build(chunks)

        # 3. Build prompt
        prompt = self.prompt_builder.build_prompt(query=question, context=context)

        # 4. Generate
        if generate:
            logger.info("[PIPELINE] Calling generator: %s", self.generator.name)
            answer = self.generator.generate(prompt)
        else:
            answer = "[Generation skipped — retrieval-only mode]"
            logger.info("[PIPELINE] Generation skipped (generate=False).")

        # 5. Verification & Trust Score
        verification_results = {}
        trust_results = {}
        if generate:
            logger.info("[PIPELINE] Running verification...")
            verification_results = self.verifier.verify(answer, context)
            trust_results = self.trust_scorer.calculate_score(verification_results, chunks, answer)

        metadata = {
            "verification": verification_results,
            "trust": trust_results,
            "query_info": query_info
        }

        result = RAGResult(
            query=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks,
            retrieval_scores=[c.similarity_score for c in chunks],
            context=context,
            prompt=prompt,
            num_chunks_found=len(chunks),
            generator_name=self.generator.name,
            metadata=metadata
        )

        logger.info(
            "[PIPELINE] Query complete. chunks=%d, generator=%s.",
            len(chunks),
            self.generator.name,
        )
        return result

    # ------------------------------------------------------------------ #
    # Document ingestion                                                   #
    # ------------------------------------------------------------------ #

    def ingest_document(self, path: str | Path) -> Dict[str, Any]:
        """Ingest a document: load → clean → chunk → embed → store.

        This is the complete document pipeline in a single call.

        Args:
            path: Path to a .txt, .pdf, or .md file.

        Returns:
            Dict with keys: document_id, document_name, num_chunks, source.

        Raises:
            FileNotFoundError, ValueError, RuntimeError: See DocumentLoader.
        """
        # 1. Load and clean
        doc = self.document_loader.load(path)
        logger.info("[PIPELINE] Document loaded: '%s'.", doc.document_name)

        # 2. Chunk
        chunks = self.chunker.chunk_document(doc)
        logger.info("[PIPELINE] Created %d chunks.", len(chunks))

        # 3. Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.embed_documents(texts)

        # 4. Build ChromaDB entries
        chunk_ids = [
            f"{doc.document_id[:16]}_{i}" for i in range(len(chunks))
        ]
        metadatas = [
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number if c.page_number is not None else "",
                "ingested_at": doc.loaded_at,
                "module": getattr(c, "module", ""),
                "topic": getattr(c, "topic", ""),
                "subtopic": getattr(c, "subtopic", ""),
                "content_type": getattr(c, "content_type", "concept"),
            }
            for c in chunks
        ]

        # 5. Store
        self.vector_store.add_chunks(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
        )
        logger.info(
            "[PIPELINE] Ingested '%s': %d chunks stored (id=%s…).",
            doc.document_name,
            len(chunks),
            doc.document_id[:8],
        )

        return {
            "document_id": doc.document_id,
            "document_name": doc.document_name,
            "num_chunks": len(chunks),
            "source": doc.source,
        }

    # ------------------------------------------------------------------ #
    # Document deletion                                                    #
    # ------------------------------------------------------------------ #

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks for a document from the retrieval index.

        IMPORTANT TERMINOLOGY:
            This is 'document deletion from the RAG knowledge base'.
            It removes data from ChromaDB ONLY.
            It does NOT modify TrustLM's model weights.
            It is NOT machine unlearning.

        Args:
            document_id: The document's SHA-256 identifier (returned by
                         ingest_document).

        Returns:
            Number of chunks deleted.
        """
        n_deleted = self.vector_store.delete_document(document_id)
        logger.info(
            "[PIPELINE] Deleted document '%s' (%d chunks removed from retrieval index).",
            document_id[:8],
            n_deleted,
        )
        return n_deleted

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the current knowledge base.

        Returns:
            Dict with keys: total_chunks, collection_name, db_path.
        """
        return {
            "total_chunks": self.vector_store.count(),
            "collection_name": self.vector_store.collection_name,
            "db_path": str(self.vector_store.db_path),
            "generator": self.generator.name,
        }
