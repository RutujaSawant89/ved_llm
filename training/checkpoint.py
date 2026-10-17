"""TrustLM Checkpointing Module.

Provides robust checkpoint serialization and deserialization for model weights,
optimizer state, learning rate schedules, and training progress metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class CheckpointManager:
    """Manages saving, loading, and discovering training checkpoints."""

    def __init__(self, checkpoint_dir: Path | str = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        filepath: Path | str,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        train_loss: float,
        val_loss: float,
        model_config: Optional[Dict[str, Any]] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save full training state to disk.

        Args:
            filepath: Destination file path.
            model: PyTorch model instance.
            optimizer: Optimizer instance.
            epoch: Current training epoch index.
            global_step: Cumulative training steps completed.
            train_loss: Most recent training loss.
            val_loss: Most recent validation loss.
            model_config: Model architecture hyperparameters.
            tokenizer_config: Tokenizer settings.
            scheduler: Optional learning rate scheduler.
            extra_metadata: Any custom research tags or metrics.

        Returns:
            Path to the saved checkpoint file.
        """
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": model_config or {},
            "tokenizer_config": tokenizer_config or {},
            "extra_metadata": extra_metadata or {},
        }

        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Atomic write pattern: save to temporary file first, then rename
        tmp_dest = dest.with_suffix(".tmp")
        torch.save(checkpoint, tmp_dest)
        if dest.exists():
            dest.unlink()
        tmp_dest.rename(dest)

        return dest

    def load_checkpoint(
        self,
        filepath: Path | str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load weights, optimizer state, and training metadata from a checkpoint.

        Args:
            filepath: Path to the checkpoint file.
            model: PyTorch model to restore weights into.
            optimizer: Optional optimizer to restore states into.
            scheduler: Optional scheduler to restore.
            device: Compute device where tensors should be mapped.

        Returns:
            Loaded checkpoint dictionary.
        """
        src = Path(filepath)
        if not src.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {src}")

        map_loc = device if device is not None else torch.device("cpu")
        checkpoint = torch.load(src, map_location=map_loc)

        # Restore model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer states if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler states if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find and return the latest checkpoint file by modification time."""
        checkpoints = sorted(
            list(self.checkpoint_dir.glob("*.pt")) + list(self.checkpoint_dir.glob("*.pth")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None
