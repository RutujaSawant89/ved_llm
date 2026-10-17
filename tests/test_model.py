"""Unit tests for Phase 4 Custom Decoder-Only Transformer (TrustLM)."""

import unittest

import torch

from model.attention import MultiHeadSelfAttention
from model.embeddings import PositionalEmbedding, TokenEmbedding
from model.feed_forward import FeedForward
from model.transformer_block import TransformerBlock
from model.trustlm import TrustLM


class TestTrustLMArchitecture(unittest.TestCase):
    """Test suite for the from-scratch custom Transformer model."""

    def setUp(self):
        # Deterministic setup
        torch.manual_seed(42)

        # Tiny configuration for CPU-friendly, fast unit testing
        self.vocab_size = 64
        self.context_length = 16
        self.embedding_dim = 32
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.0  # Zero dropout for deterministic causal testing
        self.bias = False

        self.model = TrustLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=self.bias,
            tie_weights=True,
        )
        self.model.eval()

    def test_forward_pass(self):
        """Test 1: Forward pass execution without runtime errors."""
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = self.model(input_ids)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertFalse(torch.isnan(logits).any(), "Logits contain NaN values")
        self.assertFalse(torch.isinf(logits).any(), "Logits contain Inf values")

    def test_output_shape(self):
        """Test 2: Verify output logits shape is (batch_size, sequence_length, vocab_size)."""
        test_shapes = [(1, 1), (2, 8), (3, self.context_length)]

        for b, t in test_shapes:
            input_ids = torch.randint(0, self.vocab_size, (b, t))
            with torch.no_grad():
                logits = self.model(input_ids)

            expected_shape = (b, t, self.vocab_size)
            self.assertEqual(
                logits.shape,
                expected_shape,
                f"Expected shape {expected_shape} for input (B={b}, T={t}), got {logits.shape}",
            )

    def test_causal_attention_mask(self):
        """Test 3: Verify causal masking property.

        Tokens at earlier positions must NOT be influenced by tokens at future positions.
        If we modify the token at position T-1, logits at positions 0 to T-2 must remain identical.
        """
        seq_len = 8
        # Base sequence
        seq_a = torch.randint(0, self.vocab_size, (1, seq_len))
        # Sequence B is identical to Sequence A except at the very last position
        seq_b = seq_a.clone()
        seq_b[0, -1] = (seq_a[0, -1] + 1) % self.vocab_size

        with torch.no_grad():
            logits_a = self.model(seq_a)
            logits_b = self.model(seq_b)

        # Logits at positions 0 to seq_len - 2 must be identical
        diff_prefix = torch.abs(logits_a[:, :-1, :] - logits_b[:, :-1, :]).max().item()
        self.assertAlmostEqual(
            diff_prefix,
            0.0,
            places=5,
            msg=f"Causal violation detected! Prefix logits changed when future token was altered. Max diff: {diff_prefix}",
        )

        # Logit at the modified last position should differ
        diff_last = torch.abs(logits_a[:, -1, :] - logits_b[:, -1, :]).max().item()
        self.assertGreater(
            diff_last,
            1e-5,
            "Expected logits at altered position to change, but they were identical.",
        )

    def test_attention_weights_upper_triangle(self):
        """Verify that multi-head self-attention explicitly zeros out upper triangular weights."""
        attn_layer = MultiHeadSelfAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            context_length=self.context_length,
            dropout=0.0,
            bias=False,
        )
        attn_layer.eval()

        x = torch.randn(2, 6, self.embedding_dim)
        _, weights = attn_layer(x, return_attention_weights=True)

        # weights shape: (batch_size, num_heads, seq_len, seq_len)
        for i in range(6):
            for j in range(i + 1, 6):
                upper_tri_val = weights[:, :, i, j].abs().max().item()
                self.assertEqual(
                    upper_tri_val,
                    0.0,
                    f"Attention weight at ({i}, {j}) is not zero! Causal mask breached: {upper_tri_val}",
                )

    def test_gradient_flow(self):
        """Test 4: Verify backpropagation computes gradients for all trainable parameters."""
        self.model.train()
        self.model.zero_grad()

        input_ids = torch.randint(0, self.vocab_size, (2, 8))
        logits = self.model(input_ids)
        loss = logits.sum()
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertTrue(
                param.requires_grad,
                f"Parameter {name} does not require grad",
            )
            self.assertIsNotNone(
                param.grad,
                f"Gradient is None for parameter: {name}",
            )
            self.assertFalse(
                torch.isnan(param.grad).any(),
                f"Gradient contains NaN for parameter: {name}",
            )
            grad_norm = param.grad.abs().sum().item()
            self.assertGreater(
                grad_norm,
                0.0,
                f"Zero gradient detected for parameter: {name}",
            )

    def test_parameter_count(self):
        """Test 5: Verify exact parameter count matches analytical calculation."""
        # Analytical parameter computation for tiny test config:
        # Token Embedding: V * C = 64 * 32 = 2048
        # Positional Embedding: T * C = 16 * 32 = 512
        # Per Transformer Block (bias=False):
        #   ln_1: C = 32
        #   attn.qkv_proj: C * 3C = 32 * 96 = 3072
        #   attn.out_proj: C * C = 32 * 32 = 1024
        #   ln_2: C = 32
        #   mlp.fc1: C * 4C = 32 * 128 = 4096
        #   mlp.fc2: 4C * C = 128 * 32 = 4096
        #   Block total = 32 + 3072 + 1024 + 32 + 4096 + 4096 = 12352
        # Blocks (2 layers) = 2 * 12352 = 24704
        # Final ln_f: C = 32
        # LM Head: tied to token embedding -> 0 unique parameters
        # Total parameters = 2048 + 512 + 24704 + 32 = 27296
        expected_total = 2048 + 512 + 24704 + 32
        expected_non_emb = 24704 + 32  # Transformer blocks + final LayerNorm

        self.assertEqual(self.model.num_parameters, expected_total)
        self.assertEqual(
            self.model.get_num_params(non_embedding=True),
            expected_non_emb,
        )

        # Print architecture summary for test run visibility
        self.model.print_architecture()

    def test_forward_with_targets(self):
        """Verify forward pass computes loss when targets are supplied."""
        input_ids = torch.randint(0, self.vocab_size, (2, 8))
        targets = torch.randint(0, self.vocab_size, (2, 8))

        logits, loss = self.model(input_ids, targets=targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar tensor
        self.assertGreater(loss.item(), 0.0)

    def test_subcomponents_directly(self):
        """Verify individual subcomponents: TokenEmbedding, PositionalEmbedding, FeedForward, TransformerBlock."""
        # 1. TokenEmbedding
        tok_emb = TokenEmbedding(self.vocab_size, self.embedding_dim)
        x_tok = tok_emb(torch.tensor([[1, 2, 3]]))
        self.assertEqual(x_tok.shape, (1, 3, self.embedding_dim))

        # 2. PositionalEmbedding
        pos_emb = PositionalEmbedding(self.context_length, self.embedding_dim)
        x_pos = pos_emb(3, device=torch.device("cpu"))
        self.assertEqual(x_pos.shape, (1, 3, self.embedding_dim))

        # 3. FeedForward
        ff = FeedForward(self.embedding_dim, hidden_dim=128, dropout=0.0, bias=False)
        x_ff = ff(x_tok)
        self.assertEqual(x_ff.shape, (1, 3, self.embedding_dim))

        # 4. TransformerBlock
        block = TransformerBlock(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            context_length=self.context_length,
            hidden_dim=128,
            dropout=0.0,
            bias=False,
        )
        x_block = block(x_tok)
        self.assertEqual(x_block.shape, (1, 3, self.embedding_dim))


if __name__ == "__main__":
    unittest.main()
