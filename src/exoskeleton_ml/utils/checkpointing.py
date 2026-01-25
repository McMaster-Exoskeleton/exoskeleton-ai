"""Checkpointing utilities for saving and loading model states."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Path | str,
    scheduler: Any | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path where to save the checkpoint.
        scheduler: Optional learning rate scheduler.
        metrics: Optional additional metrics to save.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load model checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        model: PyTorch model to load weights into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to load the model onto.

    Returns:
        Dictionary containing checkpoint information (epoch, loss, metrics).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"✅ Checkpoint loaded from {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "metrics": checkpoint.get("metrics", {}),
    }


def save_best_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    output_dir: Path | str,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save the best model checkpoint.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        output_dir: Directory where to save the checkpoint.
        metrics: Optional additional metrics to save.
    """
    output_dir = Path(output_dir)
    filepath = output_dir / "best_model.pt"
    save_checkpoint(model, optimizer, epoch, loss, filepath, metrics=metrics)
