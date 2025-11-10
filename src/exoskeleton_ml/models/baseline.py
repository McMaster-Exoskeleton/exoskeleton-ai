"""Baseline models for movement recognition."""

import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """Simple baseline model using LSTM."""

    def __init__(
        self,
        _input_size: int,
        _hidden_size: int = 128,
        _num_classes: int = 5,
        _num_layers: int = 2,
        _dropout: float = 0.2,
    ) -> None:
        """Initialize baseline model.

        Args:
            _input_size: Number of input features.
            _hidden_size: Hidden layer size.
            _num_classes: Number of output classes.
            _num_layers: Number of LSTM layers.
            _dropout: Dropout rate.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        pass
