"""Per-timestep linear regression baseline for joint moment estimation."""

import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    """Linear regression baseline: predicts each timestep independently (no temporal context)."""

    def __init__(self, input_size: int, output_size: int, init_std: float = 0.01) -> None:
        """Initialize linear baseline.

        Args:
            input_size: Number of input features (e.g. 28: 24 IMU + 4 angles).
            output_size: Number of output targets (e.g. 4 joint moments).
            init_std: Standard deviation for linear layer weight initialization.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear.weight.data.normal_(0, init_std)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output tensor of shape (batch, seq_len, output_size).
        """
        return self.linear(x)

    def get_effective_history(self) -> int:
        """Effective history in timesteps (1 = no temporal context)."""
        return 1
