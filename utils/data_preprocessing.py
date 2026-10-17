"""TrustLM Data Preprocessing Pipeline (Phase 1).

A robust, modular data preparation pipeline built using standard Python libraries.
Performs:
  1. Loading text files from data/raw/
  2. Normalizing whitespace (unicode normalization, tab handling, collapsing spaces)
  3. Removing malformed content (control characters, null bytes, HTML tags)
  4. Removing excessive repeated characters
  5. Removing empty lines
  6. Saving cleaned text to data/processed/ and creating data/train.txt, data/val.txt
  7. Deterministic train/validation dataset splitting
  8. Computing and reporting comprehensive dataset statistics
"""

from __future__ import annotations

import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetStatistics:
    """Statistical summary of the preprocessed dataset."""

    num_documents: int
    num_characters: int
    approx_words: int
    train_docs: int
    val_docs: int
    train_characters: int
    val_characters: int
    train_words: int
    val_words: int
    train_ratio: float

    def to_dict(self) -> Dict[str, int | float]:
        return {
            "num_documents": self.num_documents,
            "num_characters": self.num_characters,
            "approx_words": self.approx_words,
            "train_docs": self.train_docs,
            "val_docs": self.val_docs,
            "train_characters": self.train_characters,
            "val_characters": self.val_characters,
            "train_words": self.train_words,
            "val_words": self.val_words,
            "train_ratio": self.train_ratio,
        }


class TextCleaner:
    """Provides pure-Python sanitization routines for raw text corpora."""

    # Matches HTML tags like <p>, </div>, <br/>
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

    @classmethod
    def remove_malformed_content(cls, text: str) -> str:
        """Remove null bytes, unprintable/control characters, and strip HTML tags.

        Preserves valid newline characters (\n, \r) and standard tabs.
        Normalizes unicode characters to standard NFKC form.
        """
        if not text:
            return ""

        # 1. Normalize unicode to NFKC standard representation
        text = unicodedata.normalize("NFKC", text)

        # 2. Remove null bytes and carriage returns
        text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip HTML tags if present from scraped sources
        text = cls.HTML_TAG_PATTERN.sub("", text)

        # 4. Filter out unprintable/control characters (category 'C') except newline and tab
        cleaned_chars = []
        for ch in text:
            if ch in ("\n", "\t"):
                cleaned_chars.append(ch)
            elif unicodedata.category(ch).startswith("C"):
                continue
            else:
                cleaned_chars.append(ch)

        return "".join(cleaned_chars)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalize unicode spaces, convert tabs to spaces, and collapse multiple horizontal spaces.

        Preserves line breaks while trimming leading/trailing whitespace from each line.
        """
        if not text:
            return ""

        # Replace non-breaking spaces and exotic unicode spaces with standard space
        text = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", text)

        # Replace horizontal tabs with standard spaces
        text = text.replace("\t", " ")

        # Process line by line: collapse multiple spaces and strip line edges
        lines = []
        for line in text.split("\n"):
            collapsed_line = re.sub(r"[ ]{2,}", " ", line).strip()
            lines.append(collapsed_line)

        return "\n".join(lines)

    @classmethod
    def remove_excessive_repeated_chars(cls, text: str, max_repeat: int = 3) -> str:
        """Truncate characters repeated more than max_repeat consecutive times.

        Example: 'soooooooo' -> 'sooo', '!!!!!!!' -> '!!!'.
        """
        if not text or max_repeat <= 0:
            return text

        pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
        return pattern.sub(lambda m: m.group(1) * max_repeat, text)

    @classmethod
    def remove_empty_lines(cls, text: str) -> str:
        """Filter out lines that are empty or consist only of whitespace."""
        if not text:
            return ""
        lines = [line.strip() for line in text.split("\n")]
        non_empty = [line for line in lines if line]
        return "\n".join(non_empty)

    @classmethod
    def clean_text(cls, text: str, max_repeat: int = 3) -> str:
        """Run the full sequential text sanitization pipeline."""
        text = cls.remove_malformed_content(text)
        text = cls.remove_excessive_repeated_chars(text, max_repeat=max_repeat)
        text = cls.normalize_whitespace(text)
        text = cls.remove_empty_lines(text)
        return text


class DataPreprocessor:
    """Manages raw corpus ingestion, sanitization, train/val partitioning, and statistical reporting."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        raw_dirname: str = "data/raw",
        processed_dirname: str = "data/processed",
        data_dirname: str = "data",
    ) -> None:
        if base_dir is None:
            # Resolve relative to TrustLM project root (two levels up from utils/data_preprocessing.py)
            base_dir = Path(__file__).resolve().parent.parent

        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / raw_dirname
        self.processed_dir = self.base_dir / processed_dirname
        self.data_dir = self.base_dir / data_dirname

        # Ensure directory structure exists
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_documents(self, raw_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
        """Load all raw text files (.txt) from the raw directory.

        Returns a list of (filename, raw_content) tuples sorted by filename.
        """
        target_dir = Path(raw_dir) if raw_dir else self.raw_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {target_dir}")

        documents: List[Tuple[str, str]] = []
        for file_path in sorted(target_dir.glob("*.txt")):
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    documents.append((file_path.name, content))
                except Exception as exc:
                    print(f"Warning: Failed to read {file_path.name}: {exc}", file=sys.stderr)

        return documents

    def clean_documents(
        self, raw_documents: List[Tuple[str, str]], max_repeat: int = 3
    ) -> List[str]:
        """Sanitize a list of raw documents. Discards empty documents after cleaning."""
        cleaned_docs: List[str] = []
        for _, content in raw_documents:
            cleaned = TextCleaner.clean_text(content, max_repeat=max_repeat)
            if cleaned:
                cleaned_docs.append(cleaned)
        return cleaned_docs

    @staticmethod
    def split_train_val(
        documents: List[str], train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """Deterministically split documents into training and validation sets.

        Guarantees:
          - If len(documents) == 1, document is assigned to training.
          - If len(documents) > 1, both training and validation have at least 1 document.
          - Reproducibility via random seed.
        """
        if not documents:
            return [], []

        n = len(documents)
        if n == 1:
            return list(documents), []

        # Create a deterministic permutation
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)

        split_idx = int(round(n * train_ratio))
        # Ensure at least 1 validation item and at least 1 training item
        split_idx = max(1, min(n - 1, split_idx))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_docs = [documents[i] for i in train_indices]
        val_docs = [documents[i] for i in val_indices]

        return train_docs, val_docs

    def save_dataset(
        self,
        train_docs: List[str],
        val_docs: List[str],
        cleaned_docs: List[str],
        doc_delimiter: str = "\n\n",
    ) -> Dict[str, Path]:
        """Save cleaned training, validation, and full datasets to data/ and data/processed/."""
        # Join documents using a standard delimiter
        train_content = doc_delimiter.join(train_docs)
        val_content = doc_delimiter.join(val_docs)
        all_cleaned_content = doc_delimiter.join(cleaned_docs)

        # Primary paths specified in project requirements: data/train.txt & data/val.txt
        train_path = self.data_dir / "train.txt"
        val_path = self.data_dir / "val.txt"

        # Secondary processed paths in data/processed/
        proc_train_path = self.processed_dir / "train.txt"
        proc_val_path = self.processed_dir / "val.txt"
        proc_all_path = self.processed_dir / "cleaned_corpus.txt"

        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_content)
        with open(val_path, "w", encoding="utf-8") as f:
            f.write(val_content)

        with open(proc_train_path, "w", encoding="utf-8") as f:
            f.write(train_content)
        with open(proc_val_path, "w", encoding="utf-8") as f:
            f.write(val_content)
        with open(proc_all_path, "w", encoding="utf-8") as f:
            f.write(all_cleaned_content)

        return {
            "train": train_path,
            "val": val_path,
            "processed_train": proc_train_path,
            "processed_val": proc_val_path,
            "processed_all": proc_all_path,
        }

    @staticmethod
    def compute_statistics(
        cleaned_docs: List[str],
        train_docs: List[str],
        val_docs: List[str],
        train_ratio: float,
    ) -> DatasetStatistics:
        """Compute characters, approximate words, and document partition counts."""
        num_docs = len(cleaned_docs)
        total_chars = sum(len(d) for d in cleaned_docs)
        total_words = sum(len(d.split()) for d in cleaned_docs)

        train_chars = sum(len(d) for d in train_docs)
        train_words = sum(len(d.split()) for d in train_docs)

        val_chars = sum(len(d) for d in val_docs)
        val_words = sum(len(d.split()) for d in val_docs)

        return DatasetStatistics(
            num_documents=num_docs,
            num_characters=total_chars,
            approx_words=total_words,
            train_docs=len(train_docs),
            val_docs=len(val_docs),
            train_characters=train_chars,
            val_characters=val_chars,
            train_words=train_words,
            val_words=val_words,
            train_ratio=train_ratio,
        )

    @staticmethod
    def print_statistics(stats: DatasetStatistics) -> None:
        """Print formatted dataset statistics to stdout."""
        line = "=" * 65
        print(f"\n{line}")
        print(" " * 18 + "DATASET PREPARATION STATISTICS")
        print(line)
        print(f" Total Documents             : {stats.num_documents:,}")
        print(f" Total Characters            : {stats.num_characters:,}")
        print(f" Approximate Total Words     : {stats.approx_words:,}")
        print("-" * 65)
        print(
            f" Training Set ({(stats.train_docs / max(1, stats.num_documents) * 100):.1f}%)        : "
            f"{stats.train_docs:,} docs | {stats.train_characters:,} chars | {stats.train_words:,} words"
        )
        print(
            f" Validation Set ({(stats.val_docs / max(1, stats.num_documents) * 100):.1f}%)      : "
            f"{stats.val_docs:,} docs | {stats.val_characters:,} chars | {stats.val_words:,} words"
        )
        print(f"{line}\n")

    def run_pipeline(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
        max_repeat: int = 3,
        verbose: bool = True,
    ) -> Tuple[DatasetStatistics, Dict[str, Path]]:
        """Run the full end-to-end data preparation pipeline."""
        if verbose:
            print(f"Loading raw text documents from: {self.raw_dir}")

        raw_docs = self.load_raw_documents()
        if not raw_docs:
            raise ValueError(f"No .txt documents found in raw directory: {self.raw_dir}")

        if verbose:
            print(f"Loaded {len(raw_docs)} raw document(s). Cleaning and normalizing...")

        cleaned_docs = self.clean_documents(raw_docs, max_repeat=max_repeat)
        if not cleaned_docs:
            raise ValueError("All documents were empty or filtered out after cleaning.")

        train_docs, val_docs = self.split_train_val(
            cleaned_docs, train_ratio=train_ratio, seed=seed
        )

        saved_paths = self.save_dataset(train_docs, val_docs, cleaned_docs)
        if verbose:
            print(f"Saved training set to: {saved_paths['train']}")
            print(f"Saved validation set to: {saved_paths['val']}")

        stats = self.compute_statistics(cleaned_docs, train_docs, val_docs, train_ratio)
        if verbose:
            self.print_statistics(stats)

        return stats, saved_paths


def main() -> None:
    """CLI entry point for running data preprocessing."""
    preprocessor = DataPreprocessor()
    stats, paths = preprocessor.run_pipeline(train_ratio=0.8, seed=42)
    print("Phase 1 Data Preparation completed successfully.")


if __name__ == "__main__":
    main()
