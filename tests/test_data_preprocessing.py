"""Unit tests for Phase 1 Data Preparation Pipeline."""

import shutil
import tempfile
import unittest
from pathlib import Path

from utils.data_preprocessing import (
    DataPreprocessor,
    DatasetStatistics,
    TextCleaner,
)


class TestTextCleaner(unittest.TestCase):
    """Unit tests for TextCleaner sanitization routines."""

    def test_normalize_whitespace(self):
        raw = "This   has    excessive     spaces  and   tabs.\t\tRight?"
        expected = "This has excessive spaces and tabs. Right?"
        self.assertEqual(TextCleaner.normalize_whitespace(raw), expected)

        # Test unicode whitespace replacement
        unicode_spaces = "Word1\u00a0Word2\u2003Word3\u3000Word4"
        self.assertEqual(TextCleaner.normalize_whitespace(unicode_spaces), "Word1 Word2 Word3 Word4")

    def test_remove_malformed_content(self):
        # Null bytes and control characters
        raw_corrupt = "Clean text\x00 with null bytes and control chars: \x07\x08!"
        cleaned = TextCleaner.remove_malformed_content(raw_corrupt)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertNotIn("\x08", cleaned)
        self.assertIn("Clean text", cleaned)

        # HTML tags
        raw_html = "<p>This is <strong>bold</strong> text.</p><br/>"
        cleaned_html = TextCleaner.remove_malformed_content(raw_html)
        self.assertEqual(cleaned_html, "This is bold text.")

    def test_remove_excessive_repeated_chars(self):
        raw = "This is sooooooooo cool and amazing!!!!!!!!!!"
        cleaned = TextCleaner.remove_excessive_repeated_chars(raw, max_repeat=3)
        self.assertEqual(cleaned, "This is sooo cool and amazing!!!")

        # Verify normal words with double letters are untouched
        normal = "Look at the small bookkeeper in the wood."
        self.assertEqual(TextCleaner.remove_excessive_repeated_chars(normal, max_repeat=3), normal)

    def test_remove_empty_lines(self):
        raw = "Line 1\n\n\n\nLine 2\n   \n\t\nLine 3\n"
        cleaned = TextCleaner.remove_empty_lines(raw)
        self.assertEqual(cleaned, "Line 1\nLine 2\nLine 3")

    def test_full_clean_text_chain(self):
        noisy = (
            "   <html><body>   Reliable   AI   systems   are   crucial!!!!!   \n\n\n"
            "This   is    soooooooo     important.\x00\x01\n   \n"
            "Final   point.   </body></html>   "
        )
        cleaned = TextCleaner.clean_text(noisy, max_repeat=3)
        expected = "Reliable AI systems are crucial!!!\nThis is sooo important.\nFinal point."
        self.assertEqual(cleaned, expected)


class TestDataLoadingAndSplitting(unittest.TestCase):
    """Unit tests for dataset loading, splitting, and saving."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.processed_dir = self.test_dir / "processed"
        self.data_dir = self.test_dir / "data"

        self.raw_dir.mkdir()
        self.processed_dir.mkdir()
        self.data_dir.mkdir()

        # Create sample dummy files
        for i in range(10):
            sample_file = self.raw_dir / f"doc_{i:02d}.txt"
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write(f"This is document number {i}. It contains useful factual knowledge.\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_raw_documents(self):
        preprocessor = DataPreprocessor(
            base_dir=self.test_dir,
            raw_dirname="raw",
            processed_dirname="processed",
            data_dirname="data",
        )
        docs = preprocessor.load_raw_documents()
        self.assertEqual(len(docs), 10)
        self.assertEqual(docs[0][0], "doc_00.txt")
        self.assertIn("document number 0", docs[0][1])

    def test_train_val_split_proportions(self):
        dummy_docs = [f"Document {i}" for i in range(10)]
        train_docs, val_docs = DataPreprocessor.split_train_val(
            dummy_docs, train_ratio=0.8, seed=42
        )

        self.assertEqual(len(train_docs), 8)
        self.assertEqual(len(val_docs), 2)
        # Disjoint sets: no overlap
        self.assertEqual(len(set(train_docs).intersection(set(val_docs))), 0)
        # Union covers all documents
        self.assertEqual(set(train_docs).union(set(val_docs)), set(dummy_docs))

    def test_split_determinism(self):
        dummy_docs = [f"Document {i}" for i in range(20)]
        train_1, val_1 = DataPreprocessor.split_train_val(dummy_docs, train_ratio=0.7, seed=123)
        train_2, val_2 = DataPreprocessor.split_train_val(dummy_docs, train_ratio=0.7, seed=123)

        self.assertEqual(train_1, train_2)
        self.assertEqual(val_1, val_2)

    def test_split_edge_cases(self):
        # Empty list
        train_0, val_0 = DataPreprocessor.split_train_val([])
        self.assertEqual(train_0, [])
        self.assertEqual(val_0, [])

        # Single document
        train_1, val_1 = DataPreprocessor.split_train_val(["Only one document"])
        self.assertEqual(train_1, ["Only one document"])
        self.assertEqual(val_1, [])

        # Two documents: ensure 1 in train, 1 in val
        train_2, val_2 = DataPreprocessor.split_train_val(["Doc A", "Doc B"], train_ratio=0.9)
        self.assertEqual(len(train_2), 1)
        self.assertEqual(len(val_2), 1)

    def test_pipeline_end_to_end(self):
        preprocessor = DataPreprocessor(
            base_dir=self.test_dir,
            raw_dirname="raw",
            processed_dirname="processed",
            data_dirname="data",
        )
        stats, paths = preprocessor.run_pipeline(train_ratio=0.8, seed=42, verbose=False)

        # Check paths exist
        self.assertTrue(paths["train"].is_file())
        self.assertTrue(paths["val"].is_file())
        self.assertTrue(paths["processed_train"].is_file())
        self.assertTrue(paths["processed_val"].is_file())
        self.assertTrue(paths["processed_all"].is_file())

        # Verify statistics
        self.assertEqual(stats.num_documents, 10)
        self.assertEqual(stats.train_docs, 8)
        self.assertEqual(stats.val_docs, 2)
        self.assertGreater(stats.num_characters, 0)
        self.assertGreater(stats.approx_words, 0)
        self.assertEqual(stats.train_characters + stats.val_characters, stats.num_characters)


if __name__ == "__main__":
    unittest.main()
