"""Baseline models for movement recognition."""

import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """Simple baseline model using LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize baseline model.

        Args:
            input_size: Number of input features.
            hidden_size: Hidden layer size.
            num_classes: Number of output classes.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
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
