"""TrustLM RAG — Text Chunker.

Splits long document text into overlapping chunks suitable for embedding
and retrieval.

WHY OVERLAP?
    When a document is split at a fixed boundary, important information that
    spans two adjacent windows can be cut in half. Overlapping chunks ensure
    that the tail of one chunk re-appears at the head of the next, so a
    retrieval query that matches that region has at least one complete chunk
    to return.

Example (chunk_size=40, chunk_overlap=10):
    Source: "The Transformer architecture consists of attention mechanisms..."

    Chunk 0: "The Transformer architecture consists of"
    Chunk 1: "consists of attention mechanisms..."
                ^^^^^^^^^ (10-char overlap)

Chunking strategy: RecursiveCharacterTextSplitter-style.
    Priority separator list: ["\n\n", "\n", ". ", " ", ""]
    Each separator is tried in order; the first one that keeps the chunk
    within chunk_size is used.  This keeps paragraph and sentence boundaries
    intact wherever possible.

Configuration:
    chunk_size    — Target maximum characters per chunk (default: 1000).
    chunk_overlap — Characters to re-include from the previous chunk (default: 200).
    Both are read from RAGConfig so they are not hard-coded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default separator priority — mirrors LangChain's RecursiveCharacterTextSplitter
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class TextChunk:
    """A single chunk produced by the chunker.

    Attributes:
        text:          The chunk text content.
        chunk_index:   Zero-based position of this chunk within its document.
        start_char:    Character offset where the chunk starts in the source text.
        end_char:      Character offset where the chunk ends in the source text.
        page_number:   Estimated source page (for PDFs; 1-based, or None).
        document_id:   Identifier of the parent document.
        document_name: Filename of the parent document.
        source:        Absolute path to the parent file.
    """

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int]
    document_id: str
    document_name: str
    source: str
    module: str = ""
    topic: str = ""
    subtopic: str = ""
    content_type: str = "concept"


class TextChunker:
    """Split document text into overlapping chunks.

    Args:
        chunk_size:    Target maximum characters per chunk.
        chunk_overlap: Number of characters to repeat between adjacent chunks.
        separators:    Ordered list of separator strings to try.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else _DEFAULT_SEPARATORS

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def chunk_document(
        self,
        document_record: Any = None,
        # Legacy keyword arguments for backward compatibility with tests
        text: Optional[str] = None,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        source: Optional[str] = None,
        pages: Optional[List[str]] = None,
    ) -> List[TextChunk]:
        """Split a document's text into TextChunk objects.

        Accepts either a DocumentRecord object (new API) or the legacy
        keyword arguments text/document_id/document_name/source/pages.

        Returns:
            Ordered list of TextChunk objects.

        Raises:
            ValueError: If text is empty after stripping.
        """
        # ── Resolve arguments ──────────────────────────────────────────
        if document_record is not None:
            # New API: DocumentRecord object
            text = document_record.text.strip()
            document_id = document_record.document_id
            document_name = document_record.document_name
            source = document_record.source
            pages = document_record.pages
            module = getattr(document_record, "module", "")
        else:
            # Legacy API: plain keyword args (used by tests)
            if text is None:
                raise ValueError("[CHUNK] Must supply document_record or text.")
            text = text.strip()
            document_id = document_id or ""
            document_name = document_name or ""
            source = source or ""
            pages = pages or []
            module = ""
            document_record = None  # no structured slides

        if not text:
            raise ValueError(
                f"[CHUNK] Cannot chunk empty document: '{document_name}'"
            )

        chunks: List[TextChunk] = []

        # Structure-aware chunking for PPTX
        if document_record is not None and hasattr(document_record, "structured_slides") and document_record.structured_slides:
            for idx, slide in enumerate(document_record.structured_slides):
                chunks.append(
                    TextChunk(
                        text=slide.text,
                        chunk_index=idx,
                        start_char=0,
                        end_char=len(slide.text),
                        page_number=slide.slide_number,
                        document_id=document_id,
                        document_name=document_name,
                        source=source,
                        module=module,
                        topic=slide.topic,
                        subtopic=slide.subtopic,
                        content_type=slide.content_type
                    )
                )
            logger.info(
                "[CHUNK] '%s' \u2192 %d structured slide chunks.",
                document_name,
                len(chunks)
            )
            return chunks

        # Fallback to standard recursive chunking
        raw_chunks = self._split(text)

        cursor = 0  # Track position in original text for start_char / end_char

        for idx, chunk_text in enumerate(raw_chunks):
            # Find the actual position of this chunk in the source text
            start = text.find(chunk_text, cursor)
            if start == -1:
                start = cursor  # Fallback
            end = start + len(chunk_text)

            # Estimate page number from per-page text (PDFs)
            page_num = _estimate_page(chunk_text, pages) if pages else None

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    page_number=page_num,
                    document_id=document_id,
                    document_name=document_name,
                    source=source,
                    module=module
                )
            )

            cursor = max(cursor, end - self.chunk_overlap)

        logger.info(
            "[CHUNK] '%s' \u2192 %d chunks (size=%d, overlap=%d).",
            document_name,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Core splitting algorithm                                             #
    # ------------------------------------------------------------------ #

    def _split(self, text: str) -> List[str]:
        """Recursively split text using the separator priority list.

        Strategy:
          1. If text fits within chunk_size, it is a leaf — return as-is.
          2. Otherwise try each separator in order.
          3. Split on the chosen separator, then merge small pieces into
             windows of chunk_size with chunk_overlap step.
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Try separators in priority order."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in separators:
            if sep == "" or sep in text:
                parts = text.split(sep) if sep else list(text)
                merged = self._merge_splits(parts, sep)
                return merged

        # Last resort: hard-cut at chunk_size
        return self._hard_cut(text)

    def _merge_splits(self, parts: List[str], separator: str) -> List[str]:
        """Merge split parts into windows of chunk_size with chunk_overlap."""
        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(separator)

            # If adding this part exceeds chunk_size, flush current window
            if current_len + part_len > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Keep overlap: drop parts from front until we're within overlap budget
                while current_parts and current_len > self.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + len(separator)

            current_parts.append(part)
            current_len += part_len

        # Flush remaining
        if current_parts:
            chunk_text = separator.join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _hard_cut(self, text: str) -> List[str]:
        """Cut text into chunks of exactly chunk_size with overlap."""
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _estimate_page(chunk_text: str, pages: List[str]) -> Optional[int]:
    """Estimate which page (1-based) a chunk belongs to.

    Searches each page's text for the beginning of the chunk text.
    Returns the first matching page index, or None if not found.
    """
    # Use the first 100 characters of the chunk as a fingerprint
    fingerprint = chunk_text[:100].strip()
    if not fingerprint:
        return None

    for page_idx, page_text in enumerate(pages):
        if fingerprint in page_text:
            return page_idx + 1  # 1-based page number

    return None
