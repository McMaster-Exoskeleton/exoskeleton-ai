"""Data preprocessing utilities."""

import numpy as np
import torch


def preprocess_signals(
    signals: np.ndarray | torch.Tensor,
    normalize: bool = True,
    remove_mean: bool = True,
) -> torch.Tensor:
    """Preprocess signal data.

    Args:
        signals: Input signals of shape (n_samples, n_features).
        normalize: Whether to normalize signals to unit variance.
        remove_mean: Whether to remove mean from signals.

    Returns:
        Preprocessed signals as PyTorch tensor.
    """
    ...


def create_sequences(
    data: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    sequence_length: int,
    step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create overlapping sequences from continuous data.

    Args:
        data: Input data of shape (n_timesteps, n_features).
        labels: Labels of shape (n_timesteps,).
        sequence_length: Length of each sequence.
        step: Step size between sequences.

    Returns:
        Tuple of (sequences, sequence_labels).
    """
    ...
