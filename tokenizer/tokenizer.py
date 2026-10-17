"""TrustLM Tokenizer Module (Phase 2).

Provides a clean, modular BPE tokenizer interface powered by tiktoken
designed to directly integrate with the custom Transformer language model.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Union

import tiktoken


class TrustLMTokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer interface for TrustLM.

    Wraps tiktoken's high-performance BPE engine to provide encoding,
    decoding, batch processing, and special token properties for the
    custom decoder-only Transformer model.
    """

    def __init__(
        self,
        encoding_name: str = "gpt2",
        pad_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        allowed_special: Union[str, Set[str]] = "all",
    ) -> None:
        """Initialize tokenizer with underlying tiktoken encoding.

        Args:
            encoding_name: Base BPE encoding scheme ('gpt2', 'cl100k_base', etc.).
            pad_token: Token string used for batch sequence padding.
            eos_token: Token string indicating end of sequence.
            allowed_special: Set of special tokens allowed during encoding,
                or "all" to allow all known special tokens.
        """
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self._pad_token_str = pad_token
        self._eos_token_str = eos_token
        self._allowed_special = allowed_special

        # Resolve special token integer IDs
        self._eos_token_id = self._resolve_special_token_id(self._eos_token_str)
        self._pad_token_id = self._resolve_special_token_id(self._pad_token_str)

    def _resolve_special_token_id(self, token_str: str) -> int:
        """Resolve the integer token ID for a special token string."""
        try:
            ids = self._enc.encode(token_str, allowed_special="all")
            if len(ids) == 1:
                return ids[0]
            elif hasattr(self._enc, "eot_token") and self._enc.eot_token is not None:
                return self._enc.eot_token
            return ids[0] if ids else 0
        except Exception:
            if hasattr(self._enc, "eot_token") and self._enc.eot_token is not None:
                return self._enc.eot_token
            return 0

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size of the tokenizer."""
        return self._enc.n_vocab

    @property
    def pad_token(self) -> str:
        """String representation of the padding token."""
        return self._pad_token_str

    @property
    def eos_token(self) -> str:
        """String representation of the end-of-sequence token."""
        return self._eos_token_str

    @property
    def pad_token_id(self) -> int:
        """Integer ID of the padding token."""
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Integer ID of the end-of-sequence token."""
        return self._eos_token_id

    def encode(
        self,
        text: str,
        allowed_special: Optional[Union[str, Set[str]]] = None,
    ) -> List[int]:
        """Encode a string into a list of integer token IDs.

        Args:
            text: Input text string to tokenize.
            allowed_special: Override default allowed special tokens.

        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        special = self._allowed_special if allowed_special is None else allowed_special
        return self._enc.encode(text, allowed_special=special)

    def decode(self, token_ids: List[int], errors: str = "replace") -> str:
        """Decode a sequence of integer token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.
            errors: Handling strategy for decoding errors ('replace', 'ignore', etc.).

        Returns:
            Reconstructed text string.
        """
        if not token_ids:
            return ""

        return self._enc.decode(token_ids, errors=errors)

    def encode_batch(
        self,
        texts: Iterable[str],
        allowed_special: Optional[Union[str, Set[str]]] = None,
    ) -> List[List[int]]:
        """Encode a batch of strings into a list of token sequences.

        Args:
            texts: Iterable of input text strings.
            allowed_special: Override default allowed special tokens.

        Returns:
            List of lists of integer token IDs.
        """
        return [self.encode(t, allowed_special=allowed_special) for t in texts]

    def decode_batch(
        self,
        token_sequences: Iterable[List[int]],
        errors: str = "replace",
    ) -> List[str]:
        """Decode a batch of token sequences back into strings.

        Args:
            token_sequences: Iterable of token ID lists.
            errors: Handling strategy for decoding errors.

        Returns:
            List of reconstructed text strings.
        """
        return [self.decode(seq, errors=errors) for seq in token_sequences]

    def __repr__(self) -> str:
        return (
            f"TrustLMTokenizer(encoding='{self.encoding_name}', "
            f"vocab_size={self.vocab_size}, "
            f"eos_token_id={self.eos_token_id}, "
            f"pad_token_id={self.pad_token_id})"
        )


def demo() -> None:
    """Demonstrate TrustLMTokenizer and display sample tokenization metrics."""
    tokenizer = TrustLMTokenizer()
    sample_text = (
        "TrustLM is a custom decoder-only language model built from scratch in PyTorch. "
        "It integrates response verification, trust scoring, and selective unlearning."
    )

    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)

    print("=" * 70)
    print("                    TrustLM Tokenizer Demo")
    print("=" * 70)
    print(f"Vocabulary Size      : {tokenizer.vocab_size:,}")
    print(f"EOS Token            : '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"PAD Token            : '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print("-" * 70)
    print(f"Sample Text          :\n  {sample_text}")
    print("-" * 70)
    print(f"Token IDs ({len(token_ids)} tokens):\n  {token_ids}")
    print("-" * 70)
    print(f"Decoded Text         :\n  {decoded_text}")
    print("-" * 70)
    round_trip_match = sample_text == decoded_text
    print(f"Round-Trip Verified  : {round_trip_match}")
    print("=" * 70)


if __name__ == "__main__":
    demo()
