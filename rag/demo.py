"""TrustLM RAG — Command-Line Demo.

Demonstrates the complete RAG pipeline interactively.

Usage:
    python -m rag.demo

    From the TrustLM project root directory.

The demo allows you to:
  1. Add documents to the knowledge base
  2. Ask questions and get answers with source citations
  3. View retrieved chunks
  4. Delete documents from the knowledge base
  5. View knowledge base statistics

This file is also the recommended way to test the RAG pipeline manually
before running the automated test suite.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Python < 3.7 fallback

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Configure logging for demo output
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs for clean demo output
    format="%(levelname)s: %(message)s",
)

from config import get_default_config
from rag.chunker import TextChunker
from rag.context_builder import ContextBuilder
from rag.document_loader import DocumentLoader
from rag.embeddings import EmbeddingManager
from rag.generator import TrustLMGenerator
from rag.pipeline import RAGPipeline
from rag.prompt_builder import PromptBuilder
from rag.retriever import Retriever
from rag.vector_store import VectorStore

# ─────────────────────────────────────────────────────────────────── #
# Formatting helpers                                                    #
# ─────────────────────────────────────────────────────────────────── #

WIDTH = 70


def line(char: str = "=") -> str:
    return char * WIDTH


def banner(title: str) -> None:
    print(f"\n{line()}")
    print(f"  {title}")
    print(line())


def section(title: str) -> None:
    print(f"\n{line('-')}")
    print(f"  {title}")
    print(line('-'))


def wrap(text: str, indent: int = 4) -> str:
    """Word-wrap text to WIDTH characters with optional indent."""
    prefix = " " * indent
    return textwrap.fill(text, width=WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix)


# ─────────────────────────────────────────────────────────────────── #
# Pipeline builder                                                      #
# ─────────────────────────────────────────────────────────────────── #


def build_pipeline(db_path: Optional[str] = None) -> RAGPipeline:
    """Construct the RAGPipeline with persistent ChromaDB storage."""
    cfg = get_default_config()

    # Use config values for chunk size and overlap
    rag_cfg = cfg.rag  # type: ignore[attr-defined]
    effective_db_path = db_path or rag_cfg.vector_db_path

    print(f"  Knowledge base: {effective_db_path}")
    print(f"  Embedding model: {rag_cfg.embedding_model}")
    print(f"  Chunk size: {rag_cfg.chunk_size} | Overlap: {rag_cfg.chunk_overlap}")

    embedding_manager = EmbeddingManager(model_name=rag_cfg.embedding_model)
    vector_store = VectorStore(
        db_path=effective_db_path,
        collection_name=rag_cfg.collection_name,
    )
    retriever = Retriever(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        top_k=rag_cfg.top_k,
    )
    chunker = TextChunker(
        chunk_size=rag_cfg.chunk_size,
        chunk_overlap=rag_cfg.chunk_overlap,
    )
    generator = TrustLMGenerator()

    return RAGPipeline(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        generator=generator,
        chunker=chunker,
        document_loader=DocumentLoader(),
    )


# ─────────────────────────────────────────────────────────────────── #
# Demo actions                                                          #
# ─────────────────────────────────────────────────────────────────── #


def demo_add_document(pipeline: RAGPipeline) -> None:
    """Prompt the user for a file path and ingest it."""
    section("ADD DOCUMENT")
    path = input("  Enter file path (.txt / .pdf / .md / .pptx): ").strip().strip('"\'')
    if not path:
        print("  No path entered. Returning to menu.")
        return
    try:
        result = pipeline.ingest_document(path)
        print(f"\n  ✓ Document ingested successfully!")
        print(f"    Name     : {result['document_name']}")
        print(f"    Chunks   : {result['num_chunks']}")
        print(f"    ID       : {result['document_id'][:16]}…")
    except Exception as exc:
        print(f"\n  ✗ Error: {exc}")


def demo_ask_question(pipeline: RAGPipeline) -> None:
    """Ask a question and display retrieval + generation results."""
    section("ASK A QUESTION")
    question = input("  Enter your question: ").strip()
    if not question:
        print("  No question entered. Returning to menu.")
        return

    stats = pipeline.get_stats()
    if stats["total_chunks"] == 0:
        print("\n  ⚠ Knowledge base is empty. Please add a document first.")
        return

    print("\n  Retrieving relevant information…")
    result = pipeline.query(question, generate=True)

    if not result.retrieved_chunks:
        print("\n  ⚠ No relevant chunks found in the knowledge base.")
        return

    print("\n-----------------------------------------")
    print("ANSWER")
    print("-----------------------------------------")
    print(wrap(result.answer))

    print("\n-----------------------------------------")
    print("EVIDENCE")
    print("-----------------------------------------")
    for i, chunk in enumerate(result.retrieved_chunks, start=1):
        # Extract metadata
        module = getattr(chunk, "module", chunk.extra_metadata.get("module", ""))
        topic = getattr(chunk, "topic", chunk.extra_metadata.get("topic", ""))
        page = chunk.page_number or "XX"
        
        print(f"Chunk {i} — Source: {chunk.document_name}")
        if module:
            print(f"Module: {module}")
        print(f"Slide/Page: {page}")
        if topic:
            print(f"Topic: {topic}")
        print(f"Relevance: {int(chunk.similarity_score * 100)}%")
        print(wrap(chunk.text, indent=2))
        print()

    print("-----------------------------------------")
    print("VERIFICATION")
    print("-----------------------------------------")
    verification = result.metadata.get("verification", {})
    if verification.get("enabled") is False:
        print(verification.get("reason", "Local verification is unavailable."))
    elif verification:
        print(f"✓ Supported claims: {verification.get('supported', 0)}")
        print(f"⚠ Partially supported: {verification.get('partially_supported', 0)}")
        print(f"✗ Unsupported: {verification.get('unsupported', 0)}")
    else:
        print("Verification skipped or no claims extracted.")

    print("\n-----------------------------------------")
    print("TRUST SCORE")
    print("-----------------------------------------")
    trust = result.metadata.get("trust", {})
    if trust:
        print(trust.get("score", "N/A"))
        if trust.get("score") == "LOW":
            print(f"Evidence Coverage: {trust.get('evidence_coverage', 'LOW')}")
            print(f"Reason: {trust.get('reason', '')}")
    else:
        print("N/A")


def demo_delete_document(pipeline: RAGPipeline) -> None:
    """Delete a document from the knowledge base by its ID."""
    section("DELETE DOCUMENT")
    print("  Note: This removes chunks from the retrieval index ONLY.")
    print("        It is NOT machine unlearning — TrustLM's model weights")
    print("        are NOT modified.")
    doc_id = input("\n  Enter document ID (or first 16 chars): ").strip()
    if not doc_id:
        print("  No ID entered. Returning to menu.")
        return

    # Support partial IDs: if user gives only 16 chars, try to match
    # For the demo we use the full ID; for brevity we tell them to use
    # the full SHA-256 or the one returned by ingest_document.
    try:
        n = pipeline.delete_document(doc_id)
        if n > 0:
            print(f"\n  ✓ Deleted {n} chunks for document '{doc_id[:16]}…'.")
        else:
            print(f"\n  ⚠ No chunks found for document ID '{doc_id[:16]}…'.")
    except Exception as exc:
        print(f"\n  ✗ Error: {exc}")


def demo_stats(pipeline: RAGPipeline) -> None:
    """Display knowledge base statistics."""
    section("KNOWLEDGE BASE STATS")
    stats = pipeline.get_stats()
    print(f"\n  Total chunks stored : {stats['total_chunks']}")
    print(f"  Collection name     : {stats['collection_name']}")
    print(f"  Database path       : {stats['db_path']}")
    print(f"  Generator           : {stats['generator']}")


# ─────────────────────────────────────────────────────────────────── #
# Main menu                                                             #
# ─────────────────────────────────────────────────────────────────── #

MENU = """
  [1] Add document to knowledge base
  [2] Ask a question
  [3] Delete document from knowledge base
  [4] Show knowledge base stats
  [q] Quit
"""


def main() -> None:
    banner("TRUSTLM RAG DEMO")
    print("  Retrieval-Augmented Generation Pipeline\n")

    print("  Initializing pipeline…")
    try:
        pipeline = build_pipeline()
    except Exception as exc:
        print(f"\n  ✗ Failed to initialize pipeline: {exc}")
        print("    Make sure sentence-transformers and chromadb are installed:")
        print("    pip install sentence-transformers chromadb pypdf")
        sys.exit(1)

    print("\n  Pipeline ready.")
    demo_stats(pipeline)

    while True:
        print(f"\n{line('─')}")
        print(MENU, end="")
        choice = input("  Choice: ").strip().lower()

        if choice == "1":
            demo_add_document(pipeline)
        elif choice == "2":
            demo_ask_question(pipeline)
        elif choice == "3":
            demo_delete_document(pipeline)
        elif choice == "4":
            demo_stats(pipeline)
        elif choice in ("q", "quit", "exit"):
            print("\n  Goodbye.\n")
            break
        else:
            print("  Unknown option. Please enter 1, 2, 3, 4, or q.")


if __name__ == "__main__":
    main()
