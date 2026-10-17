"""Unit tests for TrustLMTokenizer (Phase 2)."""

import unittest

from tokenizer.tokenizer import TrustLMTokenizer


class TestTrustLMTokenizer(unittest.TestCase):
    """Test suite for BPE-based TrustLMTokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = TrustLMTokenizer()

    def test_vocab_size(self):
        """Verify vocabulary size is positive and matches GPT-2 BPE standard."""
        self.assertIsInstance(self.tokenizer.vocab_size, int)
        self.assertEqual(self.tokenizer.vocab_size, 50257)

    def test_special_tokens(self):
        """Verify special token properties and IDs."""
        self.assertEqual(self.tokenizer.eos_token, "<|endoftext|>")
        self.assertEqual(self.tokenizer.pad_token, "<|endoftext|>")
        self.assertEqual(self.tokenizer.eos_token_id, 50256)
        self.assertEqual(self.tokenizer.pad_token_id, 50256)

    def test_encode_returns_integers(self):
        """Verify encode produces a list of valid integer token IDs."""
        text = "Deep learning and Transformer architectures."
        token_ids = self.tokenizer.encode(text)

        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)
        for token_id in token_ids:
            self.assertIsInstance(token_id, int)
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.tokenizer.vocab_size)

    def test_round_trip_reconstruction(self):
        """Verify text -> encode -> token IDs -> decode -> reconstructed text equality."""
        test_samples = [
            "Hello, world!",
            "TrustLM is an open research project exploring reliable AI generation.",
            "Special characters: !@#$%^&*()_+-=[]{}|;':,.<>?/`~",
            "Numbers and scientific notation: 3.14159, 299792458 m/s, 1e-4.",
            "Multi-line paragraph:\nLine 1: Machine learning.\nLine 2: Selective unlearning.",
        ]

        for sample in test_samples:
            token_ids = self.tokenizer.encode(sample)
            reconstructed = self.tokenizer.decode(token_ids)
            self.assertEqual(
                reconstructed,
                sample,
                f"Round-trip failed for sample:\nOriginal: {sample}\nReconstructed: {reconstructed}",
            )

    def test_empty_string_handling(self):
        """Verify empty strings encode to empty lists and decode back to empty strings."""
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_batch_encode_and_decode(self):
        """Verify batch processing preserves order and enables round-trip reconstruction."""
        batch_texts = [
            "Document A: Scaled dot-product attention.",
            "Document B: Controlled modular memory systems.",
            "Document C: Integrated response verification.",
        ]

        batch_ids = self.tokenizer.encode_batch(batch_texts)
        self.assertEqual(len(batch_ids), len(batch_texts))

        batch_reconstructed = self.tokenizer.decode_batch(batch_ids)
        self.assertEqual(batch_reconstructed, batch_texts)

    def test_special_token_in_text(self):
        """Verify text containing endoftext token is properly encoded without crashing."""
        text_with_eot = "Prompt text <|endoftext|> Completion text"
        token_ids = self.tokenizer.encode(text_with_eot)
        self.assertIn(self.tokenizer.eos_token_id, token_ids)
        reconstructed = self.tokenizer.decode(token_ids)
        self.assertEqual(reconstructed, text_with_eot)

    def test_demonstration_print(self):
        """Demonstrate and print required metrics: vocab size, sample text, token IDs, decoded text."""
        sample_text = (
            "TrustLM: A Custom Large Language Model for Reliable AI Responses. "
            "Building decoder-only Transformer from scratch in PyTorch."
        )
        token_ids = self.tokenizer.encode(sample_text)
        decoded_text = self.tokenizer.decode(token_ids)

        print("\n" + "=" * 70)
        print("          TOKENIZER DEMONSTRATION & ROUND-TRIP VERIFICATION")
        print("=" * 70)
        print(f"Vocabulary Size      : {self.tokenizer.vocab_size:,}")
        print(f"Sample Text          :\n  '{sample_text}'")
        print(f"Token IDs ({len(token_ids)})      :\n  {token_ids}")
        print(f"Decoded Text         :\n  '{decoded_text}'")
        print(f"Reconstructed Match  : {sample_text == decoded_text}")
        print("=" * 70 + "\n")

        self.assertEqual(sample_text, decoded_text)


if __name__ == "__main__":
    unittest.main()
