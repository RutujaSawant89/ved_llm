"""TrustLM RAG — Context Builder.

Formats a list of retrieved chunks into a structured context block that
is ready to be injected into the prompt.

WHY NOT JUST CONCATENATE CHUNKS?
    Concatenation without metadata loses track of which text came from
    which document or page.  When TrustLM later needs to cite sources or
    when the Verification module needs to trace a claim back to evidence,
    the origin information must be preserved in the context string itself.

Output format (one block per chunk):
    ─────────────────────────────────────
    [Source 1]
    Document : machine_learning.pdf
    Page     : 4

    The Transformer architecture consists of...

    ─────────────────────────────────────
    [Source 2]
    Document : ai_intro.txt
    Page     : N/A

    Attention mechanisms allow the model to...

    ─────────────────────────────────────

The ContextBuilder also returns structured source_info dicts for the
pipeline response so the caller can display them separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

DIVIDER = "─" * 50


@dataclass
class SourceInfo:
    """Structured source reference for a retrieved chunk.

    Attributes:
        source_number: 1-based display index.
        document_name: Filename of the source document.
        document_id:   SHA-256 ID of the source document.
        source_path:   Absolute path to the source file.
        page_number:   Page number (1-based) or None.
        similarity:    Cosine similarity score in [0, 1].
        chunk_index:   Position of the chunk within its document.
    """

    source_number: int
    document_name: str
    document_id: str
    source_path: str
    page_number: Optional[int]
    similarity: float
    chunk_index: int

    def display(self) -> str:
        """Return a human-readable one-line source citation."""
        page = f"page {self.page_number}" if self.page_number else "page N/A"
        return (
            f"{self.source_number}. {self.document_name} — {page} "
            f"(similarity: {self.similarity:.3f})"
        )


class ContextBuilder:
    """Format retrieved chunks into a prompt-ready context string.

    Args:
        max_context_chars: Hard limit on total context characters.
                           Chunks are added in retrieval rank order until
                           the limit is reached.  None = no limit.
    """

    def __init__(self, max_context_chars: Optional[int] = 8000) -> None:
        self.max_context_chars = max_context_chars

    def build(
        self, chunks: List[RetrievedChunk]
    ) -> tuple[str, List[SourceInfo]]:
        """Build context string and structured source list from retrieved chunks.

        Args:
            chunks: Ranked list of RetrievedChunk objects (most relevant first).

        Returns:
            Tuple of:
              - context_str: Formatted multi-source context block.
              - sources:     List of SourceInfo objects (one per included chunk).
        """
        if not chunks:
            logger.warning("[CONTEXT] No chunks provided — returning empty context.")
            return "", []

        context_parts: List[str] = []
        sources: List[SourceInfo] = []
        total_chars = 0

        for idx, chunk in enumerate(chunks):
            source_num = idx + 1
            page_str = str(chunk.page_number) if chunk.page_number else "N/A"

            block = (
                f"\n{DIVIDER}\n"
                f"[Source {source_num}]\n"
                f"Document : {chunk.document_name}\n"
                f"Page     : {page_str}\n"
                f"Score    : {chunk.similarity_score:.3f}\n"
                f"\n"
                f"{chunk.text.strip()}\n"
            )

            # Respect max_context_chars limit
            if self.max_context_chars is not None:
                if total_chars + len(block) > self.max_context_chars:
                    logger.info(
                        "[CONTEXT] Context limit reached after %d sources. "
                        "Skipping remaining %d chunks.",
                        source_num - 1,
                        len(chunks) - idx,
                    )
                    break

            context_parts.append(block)
            total_chars += len(block)

            sources.append(
                SourceInfo(
                    source_number=source_num,
                    document_name=chunk.document_name,
                    document_id=chunk.document_id,
                    source_path=chunk.source,
                    page_number=chunk.page_number,
                    similarity=chunk.similarity_score,
                    chunk_index=chunk.chunk_index,
                )
            )

        context_parts.append(f"\n{DIVIDER}")
        context_str = "".join(context_parts)

        logger.info(
            "[CONTEXT] Built context from %d sources (%d chars).",
            len(sources),
            total_chars,
        )
        return context_str, sources

    def format_sources_display(self, sources: List[SourceInfo]) -> str:
        """Format source list for human-readable display.

        Args:
            sources: List of SourceInfo objects.

        Returns:
            Multi-line string with numbered source citations.
        """
        if not sources:
            return "No sources retrieved."
        lines = ["\nSources:"]
        for s in sources:
            lines.append(f"  {s.display()}")
        return "\n".join(lines)
