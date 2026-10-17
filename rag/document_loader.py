"""TrustLM RAG — Document Loader.

Loads documents from the filesystem and extracts clean plain text.
Supports: .txt, .md (built-in), .pdf (requires pypdf).

The loader is intentionally simple and secure:
  - Only reads files — never executes document content.
  - Validates file extensions before processing.
  - Returns structured DocumentRecord objects (not raw bytes).

Usage:
    loader = DocumentLoader()
    doc    = loader.load("data/knowledge_base/docs/ai_intro.pdf")
    print(doc.text[:200])
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Supported file extensions and the loader that handles them
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".pptx"}


@dataclass
class StructuredSlide:
    """A slide extracted from a PPTX file, preserving structure."""
    slide_number: int
    topic: str
    subtopic: str
    content_type: str
    text: str

@dataclass
class DocumentRecord:
    """Structured container for a loaded document.

    Attributes:
        document_id:   Stable SHA-256 hash of the file path (used as ChromaDB key).
        document_name: Filename without directory prefix.
        source:        Absolute path to the original file.
        text:          Cleaned plain text extracted from the document.
        page_count:    Number of pages (PDF only; 1 for flat text files).
        loaded_at:     ISO-8601 timestamp of when the document was loaded.
        extra_metadata: Any additional key-value pairs (e.g. author, title).
    """

    document_id: str
    document_name: str
    source: str
    text: str
    page_count: int = 1
    loaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    extra_metadata: dict = field(default_factory=dict)
    # For PDFs we also keep per-page text so chunking can attach page numbers.
    pages: List[str] = field(default_factory=list)
    # For PPTX, we keep fully structured slides
    structured_slides: List[StructuredSlide] = field(default_factory=list)
    module: str = ""


class DocumentLoader:
    """Load and clean documents from disk.

    The class is stateless — instantiate once and call load() repeatedly.
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self, path: str | Path) -> DocumentRecord:
        """Load a single document and return a DocumentRecord.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            A DocumentRecord with clean text and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file extension is not supported.
            RuntimeError:      If the file cannot be read (e.g. corrupt PDF).
        """
        filepath = Path(path).resolve()

        if not filepath.exists():
            raise FileNotFoundError(f"[LOAD] Document not found: {filepath}")

        ext = filepath.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"[LOAD] Unsupported file type '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        # Derive a stable document_id from the file path
        document_id = _hash_path(filepath)
        document_name = filepath.name

        structured_slides = []
        module_match = re.search(r'(M[-_]?\d+|Module[-_]?\s*\d+)', document_name, re.IGNORECASE)
        module = module_match.group(0).upper().replace("_", "-") if module_match else "Unknown"

        logger.info("[INGEST] Loading document: %s (id=%s)", document_name, document_id[:8])

        if ext == ".pdf":
            text, pages = self._load_pdf(filepath)
            page_count = len(pages) if pages else 1
        elif ext == ".pptx":
            text, pages, structured_slides = self._load_pptx(filepath)
            page_count = len(pages) if pages else 1
        else:
            # .txt and .md are treated identically
            text, pages = self._load_text(filepath)
            page_count = 1

        # Final text sanity check
        if not text.strip():
            raise RuntimeError(
                f"[LOAD] Document '{document_name}' produced empty text after extraction."
            )

        logger.info(
            "[INGEST] Loaded '%s' — %d chars, %d page(s).",
            document_name,
            len(text),
            page_count,
        )

        return DocumentRecord(
            document_id=document_id,
            document_name=document_name,
            source=str(filepath),
            text=text,
            page_count=page_count,
            pages=pages,
            structured_slides=structured_slides,
            module=module
        )

    def load_many(self, paths: List[str | Path]) -> List[DocumentRecord]:
        """Load multiple documents, skipping failures with a warning.

        Args:
            paths: List of file paths.

        Returns:
            List of successfully loaded DocumentRecord objects.
        """
        records: List[DocumentRecord] = []
        for p in paths:
            try:
                records.append(self.load(p))
            except Exception as exc:
                logger.warning("[INGEST] Skipping '%s': %s", p, exc)
        return records

    # ------------------------------------------------------------------ #
    # Private loaders                                                      #
    # ------------------------------------------------------------------ #

    def _load_text(self, filepath: Path) -> tuple[str, List[str]]:
        """Read a plain-text or Markdown file."""
        try:
            raw = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"[LOAD] Cannot read '{filepath.name}': {exc}") from exc

        cleaned = _clean_text(raw)
        # Treat the whole file as a single "page"
        return cleaned, [cleaned]

    def _load_pdf(self, filepath: Path) -> tuple[str, List[str]]:
        """Extract text from a PDF file using pypdf.

        pypdf is only imported here so the rest of the RAG system works
        even if pypdf is not installed (an ImportError is raised with a
        helpful installation message).
        """
        try:
            import pypdf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "[LOAD] PDF support requires 'pypdf'. "
                "Install it with:  pip install pypdf"
            ) from exc

        pages: List[str] = []
        try:
            reader = pypdf.PdfReader(str(filepath))
            for page in reader.pages:
                raw_page = page.extract_text() or ""
                pages.append(_clean_text(raw_page))
        except Exception as exc:
            raise RuntimeError(
                f"[LOAD] Failed to parse PDF '{filepath.name}': {exc}"
            ) from exc

        full_text = "\n\n".join(p for p in pages if p.strip())
        return full_text, pages

    def _load_pptx(self, filepath: Path) -> tuple[str, List[str], List[StructuredSlide]]:
        """Extract structured slides from a PPTX file."""
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ImportError(
                "[LOAD] PPTX support requires 'python-pptx'. "
                "Install it with:  pip install python-pptx"
            ) from exc

        pages: List[str] = []
        structured_slides: List[StructuredSlide] = []
        
        try:
            prs = Presentation(str(filepath))
            for i, slide in enumerate(prs.slides):
                slide_num = i + 1
                topic = ""
                text_pieces = []
                
                if slide.shapes.title and slide.shapes.title.has_text_frame:
                    topic = _clean_text(slide.shapes.title.text)
                
                for shape in slide.shapes:
                    if shape.has_text_frame and shape != slide.shapes.title:
                        text_pieces.append(_clean_text(shape.text))
                
                raw_page = "\n\n".join(p for p in text_pieces if p)
                if not raw_page.strip() and topic:
                    raw_page = topic
                    
                if not raw_page.strip():
                    continue
                
                # Heuristic for content type based on topic
                content_type = "concept"
                t_lower = topic.lower()
                if "vs" in t_lower or "compare" in t_lower or "difference" in t_lower:
                    content_type = "comparison"
                elif "example" in t_lower:
                    content_type = "example"
                elif "advantage" in t_lower or "benefit" in t_lower:
                    content_type = "advantage"
                elif "disadvantage" in t_lower or "drawback" in t_lower:
                    content_type = "disadvantage"
                elif "application" in t_lower or "use case" in t_lower:
                    content_type = "application"
                elif "architecture" in t_lower:
                    content_type = "architecture"
                elif "flow" in t_lower or "process" in t_lower or "step" in t_lower:
                    content_type = "process"
                elif "define" in t_lower or "definition" in t_lower:
                    content_type = "definition"
                
                # Subtopic heuristic: first bold/large text or simply empty if none
                subtopic = "" 
                
                pages.append(raw_page)
                structured_slides.append(StructuredSlide(
                    slide_number=slide_num,
                    topic=topic or f"Slide {slide_num}",
                    subtopic=subtopic,
                    content_type=content_type,
                    text=raw_page
                ))
        except Exception as exc:
            raise RuntimeError(f"[LOAD] Failed to parse PPTX '{filepath.name}': {exc}") from exc
            
        full_text = "\n\n".join(p for p in pages if p.strip())
        return full_text, pages, structured_slides


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _hash_path(filepath: Path) -> str:
    """Compute a deterministic SHA-256 hex digest from the absolute path string."""
    return hashlib.sha256(str(filepath).encode()).hexdigest()


def _clean_text(text: str) -> str:
    """Normalize and clean raw extracted text.

    Steps:
      1. Unicode NFC normalization (collapses ligatures, etc.).
      2. Replace non-breaking spaces and zero-width chars.
      3. Collapse excessive whitespace while preserving paragraph breaks.
      4. Strip leading/trailing whitespace.
    """
    # 1. Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # 2. Replace problem characters
    text = text.replace("\u00a0", " ")   # non-breaking space
    text = text.replace("\u200b", "")    # zero-width space
    text = text.replace("\ufeff", "")    # BOM

    # 3. Normalize line endings and collapse excess blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace

    # 4. Strip
    return text.strip()
