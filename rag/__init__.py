"""TrustLM RAG (Retrieval-Augmented Generation) Package.

This package implements a complete, modular RAG pipeline for TrustLM.

Pipeline overview:
    USER QUERY
        ↓
    QUERY EMBEDDING  (sentence-transformers: all-MiniLM-L6-v2)
        ↓
    VECTOR DATABASE SEARCH  (ChromaDB)
        ↓
    TOP-K RELEVANT CHUNKS
        ↓
    CONTEXT BUILDER
        ↓
    PROMPT CONSTRUCTION
        ↓
    GENERATION MODEL  (TrustLMGenerator or fallback)
        ↓
    FINAL RESPONSE + SOURCE EVIDENCE

IMPORTANT NOTE on embeddings:
    The embeddings in this package are *semantic retrieval embeddings* produced
    by all-MiniLM-L6-v2 and stored in ChromaDB.  They are completely separate
    from the token embeddings (TokenEmbedding / PositionalEmbedding) inside
    the TrustLM Transformer, which map discrete token IDs to dense vectors for
    language modelling.  Do not confuse the two.

Modules:
    document_loader  — Load .txt / .pdf / .md files and extract clean text
    chunker          — Split text into overlapping chunks
    embeddings       — EmbeddingManager wrapping all-MiniLM-L6-v2
    vector_store     — ChromaDB-based persistent vector store
    retriever        — Convert query → embedding → top-k chunk retrieval
    context_builder  — Format retrieved chunks into a readable context block
    prompt_builder   — Build the final prompt sent to the generator
    generator        — BaseGenerator / TrustLMGenerator interface
    pipeline         — RAGPipeline: the central orchestration class
    demo             — Command-line demo (python -m rag.demo)
"""

from rag.pipeline import RAGPipeline
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.document_loader import DocumentLoader
from rag.chunker import TextChunker

__all__ = [
    "RAGPipeline",
    "EmbeddingManager",
    "VectorStore",
    "Retriever",
    "DocumentLoader",
    "TextChunker",
]
