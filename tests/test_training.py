"""Unit tests for Phase 5 Pretraining Pipeline and Training Subsystem."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from model.trustlm import TrustLM
from tokenizer.tokenizer import TrustLMTokenizer
from training.checkpoint import CheckpointManager
from training.device import get_device_info, get_training_device
from training.trainer import (
    AutoregressiveDataset,
    TrustLMTrainer,
    verify_dataset_safety,
)


class TestTrainingPipeline(unittest.TestCase):
    """Unit tests for training device discovery, datasets, optimization, and checkpointing."""

    def setUp(self):
        torch.manual_seed(42)
        self.test_dir = Path(tempfile.mkdtemp())

        # Tiny model config for fast CPU testing
        self.vocab_size = 64
        self.context_length = 16
        self.embedding_dim = 32
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.0

        self.model = TrustLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=False,
            tie_weights=True,
        )

        # Synthetic token IDs
        self.dummy_tokens = [i % self.vocab_size for i in range(100)]

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_device_selection(self):
        """Test device discovery and telemetry reporting."""
        info = get_device_info()
        self.assertIn("device", info)
        self.assertIn("is_cuda", info)
        self.assertIn("pytorch_version", info)

        dev_cpu = get_training_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

    def test_autoregressive_dataset_shift(self):
        """Test that inputs and targets are shifted by exactly 1 position."""
        tokens = list(range(32))
        ds = AutoregressiveDataset(
            token_ids=tokens,
            context_length=8,
            vocab_size=64,
            stride=8,
        )
        self.assertGreater(len(ds), 0)

        x, y = ds[0]
        self.assertEqual(x.shape, (8,))
        self.assertEqual(y.shape, (8,))

        # Invariant: targets y[0:-1] == inputs x[1:]
        self.assertTrue(torch.equal(x[1:], y[:-1]))
        # First target should be second input token
        self.assertEqual(y[0].item(), x[1].item())

    def test_single_training_step_cpu(self):
        """Test forward pass, loss calculation, and backward pass on CPU."""
        self.model.train()
        x = torch.randint(0, self.vocab_size, (2, self.context_length))
        y = torch.randint(0, self.vocab_size, (2, self.context_length))

        logits, loss = self.model(x, targets=y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing gradient on {name}")

    def test_optimizer_weight_update(self):
        """Test that optimizer step updates model weights."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        initial_param = self.model.blocks[0].mlp.fc1.weight.clone().detach()

        x = torch.randint(0, self.vocab_size, (2, self.context_length))
        y = torch.randint(0, self.vocab_size, (2, self.context_length))

        optimizer.zero_grad()
        _, loss = self.model(x, targets=y)
        loss.backward()
        optimizer.step()

        updated_param = self.model.blocks[0].mlp.fc1.weight.detach()
        param_diff = (updated_param - initial_param).abs().sum().item()
        self.assertGreater(param_diff, 0.0, "Weights did not update after optimizer step")

    def test_validation_and_perplexity_calculation(self):
        """Test validation computation and verify perplexity is calculated strictly from val_loss."""
        val_tokens = [i % self.vocab_size for i in range(50)]
        val_ds = AutoregressiveDataset(val_tokens, self.context_length, self.vocab_size)
        train_ds = AutoregressiveDataset(self.dummy_tokens, self.context_length, self.vocab_size)

        trainer = TrustLMTrainer(
            model=self.model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            learning_rate=1e-3,
            batch_size=2,
            max_iters=5,
            device=torch.device("cpu"),
            checkpoint_dir=self.test_dir / "checkpoints",
            plots_dir=self.test_dir / "plots",
        )

        val_loss, ppl = trainer.validate()
        self.assertGreater(val_loss, 0.0)
        # Verify PPL = exp(val_loss)
        expected_ppl = math.exp(val_loss)
        self.assertAlmostEqual(ppl, expected_ppl, places=4)

    def test_checkpoint_save_and_load(self):
        """Test checkpoint serialization and exact state restoration."""
        ckpt_dir = self.test_dir / "checkpoints"
        manager = CheckpointManager(ckpt_dir)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        saved_path = manager.save_checkpoint(
            filepath=ckpt_dir / "test_ckpt.pt",
            model=self.model,
            optimizer=optimizer,
            epoch=1,
            global_step=10,
            train_loss=2.5,
            val_loss=2.7,
            model_config=self.model.get_parameter_summary(),
        )
        self.assertTrue(saved_path.is_file())

        # Perturb model weights
        original_fc1 = self.model.blocks[0].mlp.fc1.weight.clone()
        with torch.no_grad():
            self.model.blocks[0].mlp.fc1.weight.add_(1.0)

        # Restore from checkpoint
        loaded = manager.load_checkpoint(saved_path, model=self.model, optimizer=optimizer)
        self.assertEqual(loaded["global_step"], 10)
        self.assertEqual(loaded["train_loss"], 2.5)

        # Verify weights match original exactly
        restored_fc1 = self.model.blocks[0].mlp.fc1.weight
        self.assertTrue(torch.equal(original_fc1, restored_fc1))

    def test_trainer_smoke_loop_and_plots(self):
        """Test end-to-end 3-step training loop, metrics history, and plot generation."""
        train_ds = AutoregressiveDataset(self.dummy_tokens, self.context_length, self.vocab_size)
        val_ds = AutoregressiveDataset(self.dummy_tokens, self.context_length, self.vocab_size)

        plots_dir = self.test_dir / "plots"
        trainer = TrustLMTrainer(
            model=self.model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            learning_rate=1e-3,
            batch_size=2,
            max_iters=3,
            eval_interval=2,
            log_interval=1,
            device=torch.device("cpu"),
            checkpoint_dir=self.test_dir / "checkpoints",
            plots_dir=plots_dir,
        )

        results = trainer.train(max_steps=3, num_epochs=1)
        self.assertEqual(results["global_step"], 3)
        self.assertGreater(results["tokens_trained"], 0)

        # Verify plot files exist
        self.assertTrue((plots_dir / "training_loss.png").is_file())
        self.assertTrue((plots_dir / "validation_loss.png").is_file())


if __name__ == "__main__":
    unittest.main()
