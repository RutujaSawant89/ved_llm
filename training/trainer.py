"""TrustLM Pretraining Pipeline and Trainer.

Encapsulates:
  - Autoregressive next-token prediction
  - AdamW optimization with cosine learning rate scheduling
  - Gradient accumulation and gradient clipping
  - GPU-friendly non-blocking tensor transfers and pinned memory
  - Periodic validation and perplexity estimation: PPL = exp(val_loss)
  - Checkpointing (best and latest)
  - Loss curve plotting with Matplotlib and Seaborn
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TrainingConfig
from model.trustlm import TrustLM
from tokenizer.tokenizer import TrustLMTokenizer
from training.checkpoint import CheckpointManager
from training.device import get_training_device, print_device_summary


class AutoregressiveDataset(Dataset):
    """Autoregressive text dataset for causal next-token prediction.

    Converts a continuous token sequence into consecutive (input, target) chunks
    shifted by 1 token position:
        inputs  = tokens[i : i + context_length]
        targets = tokens[i + 1 : i + context_length + 1]
    """

    def __init__(
        self,
        token_ids: List[int],
        context_length: int,
        vocab_size: int,
        stride: Optional[int] = None,
    ) -> None:
        if not token_ids:
            raise ValueError("Token sequence cannot be empty.")
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer.")

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.stride = stride if stride is not None else context_length

        # Verify all token IDs are within [0, vocab_size)
        min_id, max_id = min(token_ids), max(token_ids)
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                f"Token IDs out of vocabulary bounds [0, {vocab_size}): min={min_id}, max={max_id}"
            )

        # Pad tokens if total length is less than context_length + 1
        needed = context_length + 1
        if len(token_ids) < needed:
            # Repeat/pad with last token or 0 to meet minimum sequence length
            pad_id = token_ids[-1] if token_ids else 0
            token_ids = token_ids + [pad_id] * (needed - len(token_ids))

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        # Build chunk indices
        self.indices: List[int] = []
        max_start = len(self.tokens) - context_length - 1
        for i in range(0, max_start + 1, self.stride):
            self.indices.append(i)

        if not self.indices:
            self.indices.append(0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        chunk = self.tokens[start : start + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def verify_dataset_safety(
    filepath: Path | str,
    tokenizer: TrustLMTokenizer,
    vocab_size: int,
    context_length: int,
) -> List[int]:
    """Verify data safety conditions before training.

    Ensures:
      - Dataset file exists
      - Dataset is not empty
      - Tokenizer successfully processes text
      - All token IDs fall within [0, vocab_size)
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Dataset file is empty: {path}")

    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError(f"Tokenizer produced zero tokens for dataset: {path}")

    min_id, max_id = min(token_ids), max(token_ids)
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Data safety failure: Token ID {max_id} exceeds vocab_size {vocab_size} in {path}"
        )

    return token_ids


class TrustLMTrainer:
    """Orchestrates model pretraining, evaluation, and checkpointing."""

    def __init__(
        self,
        model: TrustLM,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        learning_rate: float = 3e-4,
        min_lr: float = 3e-5,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        max_iters: int = 5000,
        warmup_iters: int = 200,
        eval_interval: int = 250,
        eval_iters: int = 50,
        log_interval: int = 50,
        device: Optional[torch.device] = None,
        checkpoint_dir: Path | str = "checkpoints",
        plots_dir: Path | str = "evaluation/plots",
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.lr = learning_rate
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.grad_clip = gradient_clip
        self.batch_size = batch_size
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.log_interval = log_interval

        # Device assignment
        self.device = device if device is not None else get_training_device()
        self.is_cuda = self.device.type == "cuda"
        self.model.to(self.device)

        # Output paths
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer: AdamW
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # DataLoader setup with GPU memory optimization
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.is_cuda,
            drop_last=(len(self.train_dataset) > self.batch_size),
        )

        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=self.is_cuda,
                drop_last=False,
            )

        # Training telemetry history
        self.history: Dict[str, List[Any]] = {
            "steps": [],
            "train_losses": [],
            "eval_steps": [],
            "val_losses": [],
            "perplexities": [],
            "learning_rates": [],
            "tokens_processed": [],
        }

        self.global_step = 0
        self.tokens_trained = 0
        self.best_val_loss = float("inf")

    def get_lr(self, step: int) -> float:
        """Compute cosine annealed learning rate with linear warmup."""
        if step < self.warmup_iters:
            # Linear warmup
            return self.lr * (step + 1) / max(1, self.warmup_iters)
        if step > self.max_iters:
            return self.min_lr

        # Cosine decay
        decay_ratio = (step - self.warmup_iters) / max(1, (self.max_iters - self.warmup_iters))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def validate(self) -> Tuple[float, float]:
        """Perform validation and compute perplexity strictly from validation loss.

        Returns:
            Tuple of (validation_loss, perplexity).
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return float("nan"), float("nan")

        self.model.eval()
        total_loss = 0.0
        batches_evaluated = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= self.eval_iters:
                    break
                x = x.to(self.device, non_blocking=self.is_cuda)
                y = y.to(self.device, non_blocking=self.is_cuda)

                _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                batches_evaluated += 1

        self.model.train()

        val_loss = total_loss / max(1, batches_evaluated)
        # Perplexity = exp(validation_loss) (clamped to avoid float overflow)
        perplexity = math.exp(min(val_loss, 100.0))

        return val_loss, perplexity

    def train(
        self,
        max_steps: Optional[int] = None,
        num_epochs: int = 1,
    ) -> Dict[str, Any]:
        """Execute the pretraining loop.

        Args:
            max_steps: Optional hard stop on training step count.
            num_epochs: Number of full passes through the training dataset.

        Returns:
            Dictionary containing final metrics and telemetry history.
        """
        target_steps = max_steps if max_steps is not None else self.max_iters
        self.model.train()

        pbar = tqdm(total=target_steps, desc="TrustLM Pretraining", unit="step")
        running_loss = 0.0
        running_count = 0

        # Initial validation before training
        initial_val_loss, initial_ppl = self.validate()
        if not math.isnan(initial_val_loss):
            self.history["eval_steps"].append(0)
            self.history["val_losses"].append(initial_val_loss)
            self.history["perplexities"].append(initial_ppl)

        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.global_step >= target_steps:
                    break

                # 1. Update learning rate according to cosine schedule
                lr = self.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # 2. Transfer batch to compute device with non-blocking transfers
                x = x.to(self.device, non_blocking=self.is_cuda)
                y = y.to(self.device, non_blocking=self.is_cuda)

                # 3. Forward pass
                _, loss = self.model(x, targets=y)

                # 4. Normalize loss for gradient accumulation
                loss_scaled = loss / self.gradient_accumulation_steps
                loss_scaled.backward()

                running_loss += loss.item()
                running_count += 1
                self.tokens_trained += x.numel()

                # 5. Optimizer step on accumulation boundaries
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    pbar.update(1)

                    # 6. Logging
                    avg_train_loss = running_loss / max(1, running_count)
                    if self.global_step % self.log_interval == 0:
                        self.history["steps"].append(self.global_step)
                        self.history["train_losses"].append(avg_train_loss)
                        self.history["learning_rates"].append(lr)
                        self.history["tokens_processed"].append(self.tokens_trained)

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_train_loss:.4f}",
                                "lr": f"{lr:.2e}",
                                "tokens": f"{self.tokens_trained:,}",
                            }
                        )

                    # 7. Periodic Validation
                    if self.global_step % self.eval_interval == 0:
                        val_loss, ppl = self.validate()
                        self.history["eval_steps"].append(self.global_step)
                        self.history["val_losses"].append(val_loss)
                        self.history["perplexities"].append(ppl)

                        # Save best checkpoint
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.checkpoint_manager.save_checkpoint(
                                filepath=self.checkpoint_manager.checkpoint_dir / "best_checkpoint.pt",
                                model=self.model,
                                optimizer=self.optimizer,
                                epoch=epoch,
                                global_step=self.global_step,
                                train_loss=avg_train_loss,
                                val_loss=val_loss,
                                model_config=self.model.get_parameter_summary(),
                            )

            if self.global_step >= target_steps:
                break

        pbar.close()

        # Final validation
        final_val_loss, final_ppl = self.validate()
        final_train_loss = running_loss / max(1, running_count)

        # Save latest checkpoint
        self.checkpoint_manager.save_checkpoint(
            filepath=self.checkpoint_manager.checkpoint_dir / "latest_checkpoint.pt",
            model=self.model,
            optimizer=self.optimizer,
            epoch=num_epochs,
            global_step=self.global_step,
            train_loss=final_train_loss,
            val_loss=final_val_loss if not math.isnan(final_val_loss) else 0.0,
            model_config=self.model.get_parameter_summary(),
        )

        # Generate loss curves
        self.plot_loss_curves()

        return {
            "global_step": self.global_step,
            "tokens_trained": self.tokens_trained,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_perplexity": final_ppl,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

    def plot_loss_curves(self) -> Tuple[Path, Path]:
        """Generate and save training and validation loss curves using Seaborn and Matplotlib."""
        train_path = self.plots_dir / "training_loss.png"
        val_path = self.plots_dir / "validation_loss.png"

        # 1. Training Loss Plot
        plt.figure()
        if self.history["steps"] and self.history["train_losses"]:
            sns.lineplot(x=self.history["steps"], y=self.history["train_losses"])
        plt.xlabel("Step")
        plt.ylabel("Training Loss")
        plt.title("TrustLM Training Loss")
        plt.savefig(train_path)
        plt.close()

        # 2. Validation Loss Plot
        plt.figure()
        if self.history["eval_steps"] and self.history["val_losses"]:
            # Filter out NaNs if any
            valid_pairs = [
                (s, v)
                for s, v in zip(self.history["eval_steps"], self.history["val_losses"])
                if not math.isnan(v)
            ]
            if valid_pairs:
                x_vals, y_vals = zip(*valid_pairs)
                sns.lineplot(x=list(x_vals), y=list(y_vals))
        plt.xlabel("Step")
        plt.ylabel("Validation Loss")
        plt.title("TrustLM Validation Loss")
        plt.savefig(val_path)
        plt.close()

        return train_path, val_path
