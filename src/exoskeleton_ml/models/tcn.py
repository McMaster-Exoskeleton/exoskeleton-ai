"""
Temporal Convolutional Network (TCN) for joint moment estimation.

This implementation is adapted from the paper:
"Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation"

Original implementation: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
Original License: MIT License
Copyright (c) 2018 CMU Locus Lab
"""


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Removes trailing padding from the sequence to ensure causal convolutions.

    Causal convolutions ensure that the output at time t only depends on
    inputs up to time t, not future inputs. This is achieved by applying
    padding to the left side and chomping (removing) from the right side.
    """

    def __init__(self, chomp_size: int):
        """Initialize Chomp1d layer.

        Args:
            chomp_size: Number of elements to remove from the end of the sequence.
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove trailing padding.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor with chomp_size elements removed from the end.
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single temporal block with residual connection.

    Each block consists of two dilated causal convolutions with normalization,
    activation, and dropout, followed by a residual connection.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        dropout_type: str = "Dropout",
        activation: str = "ReLU",
        norm: str = "weight_norm",
    ):
        """Initialize TemporalBlock.

        Args:
            n_inputs: Number of input channels.
            n_outputs: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            dilation: Dilation factor for the convolution.
            padding: Padding size (should be (kernel_size-1) * dilation for causal).
            dropout: Dropout probability.
            dropout_type: Type of dropout ('Dropout' or 'Dropout2d').
            activation: Activation function name (e.g., 'ReLU', 'GELU').
            norm: Normalization type ('weight_norm', 'BatchNorm1d', 'LayerNorm').
        """
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # Second convolution
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # Chomp layers to remove padding
        self.chomp1 = Chomp1d(padding)
        self.chomp2 = Chomp1d(padding)

        # Activation functions
        self.af1 = getattr(nn, activation)()
        self.af2 = getattr(nn, activation)()

        # Dropout layers
        self.dropout1 = getattr(nn, dropout_type)(dropout)
        self.dropout2 = getattr(nn, dropout_type)(dropout)

        # Build the network based on normalization type
        if norm == "weight_norm":
            # Apply weight normalization to convolutions
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

            self.net = nn.Sequential(
                self.conv1,
                self.chomp1,
                self.af1,
                self.dropout1,
                self.conv2,
                self.chomp2,
                self.af2,
                self.dropout2,
            )
        else:
            # Use batch/layer normalization
            self.norm1 = getattr(nn, norm)(n_outputs)
            self.norm2 = getattr(nn, norm)(n_outputs)

            self.net = nn.Sequential(
                self.conv1,
                self.norm1,
                self.chomp1,
                self.af1,
                self.dropout1,
                self.conv2,
                self.norm2,
                self.chomp2,
                self.af2,
                self.dropout2,
            )

        # Residual connection (1x1 conv if input/output channels differ)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        # Final activation after residual addition
        self.af = getattr(nn, activation)()

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize convolutional weights with small random values."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor of shape (batch, channels, seq_len).
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.af(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network: stacks multiple TemporalBlocks.

    Each block has an exponentially increasing dilation factor, allowing
    the network to capture long-range dependencies efficiently.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        dropout_type: str = "Dropout",
        activation: str = "ReLU",
        norm: str = "weight_norm",
    ):
        """Initialize TemporalConvNet.

        Args:
            num_inputs: Number of input features.
            num_channels: List of channel sizes for each layer.
            kernel_size: Kernel size for all convolutions.
            dropout: Dropout probability.
            dropout_type: Type of dropout ('Dropout' or 'Dropout2d').
            activation: Activation function name.
            norm: Normalization type.
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    dropout_type=dropout_type,
                    activation=activation,
                    norm=norm,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all temporal blocks.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor of shape (batch, channels, seq_len).
        """
        return self.network(x)


class TCN(nn.Module):
    """Complete TCN model for biological joint moment estimation.

    This model takes IMU and joint angle data as input and predicts
    biological joint moments at each timestep.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
        eff_hist: int,
        spatial_dropout: bool = False,
        activation: str = "ReLU",
        norm: str = "weight_norm",
        center: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ):
        """Initialize TCN model.

        Args:
            input_size: Number of input features (28: 24 IMU + 4 angles).
            output_size: Number of output features (4: joint moments).
            num_channels: List of channel sizes for TCN layers.
            kernel_size: Kernel size for convolutions.
            dropout: Dropout probability.
            eff_hist: Effective history (receptive field) in timesteps.
            spatial_dropout: Whether to use spatial dropout (Dropout2d).
            activation: Activation function name.
            norm: Normalization type.
            center: Optional normalization mean (for compatibility, prefer dataset norm).
            scale: Optional normalization std (for compatibility, prefer dataset norm).
        """
        super().__init__()

        # Create TCN backbone
        self.dropout_type = "Dropout2d" if spatial_dropout else "Dropout"
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dropout_type=self.dropout_type,
            activation=activation,
            norm=norm,
        )

        # Output linear layer
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

        # Store effective history
        self.eff_hist = eff_hist

        # Normalization parameters (optional, prefer dataset normalization)
        self.register_buffer(
            "center", center if center is not None else torch.zeros(input_size)
        )
        self.register_buffer(
            "scale", scale if scale is not None else torch.ones(input_size)
        )

    def init_weights(self) -> None:
        """Initialize linear layer weights."""
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output tensor of shape (batch, seq_len, output_size).
        """
        # Input shape: (batch, seq_len, features)
        # TCN expects: (batch, features, seq_len)

        # Transpose to (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Optional normalization (prefer dataset normalization instead)
        if self.center is not None and self.scale is not None:
            # Expand dimensions for broadcasting: (1, features, 1)
            center = self.center.view(1, -1, 1)
            scale = self.scale.view(1, -1, 1)
            x = (x - center) / scale

        # Forward pass through TCN
        out = self.tcn(x)  # (batch, num_channels[-1], seq_len)

        # Transpose back to (batch, seq_len, num_channels[-1])
        out = out.transpose(1, 2)

        # Apply linear layer to get final predictions
        out = self.linear(out)  # (batch, seq_len, output_size)

        return out

    def get_effective_history(self) -> int:
        """Get the effective history (receptive field) of the network.

        Returns:
            Number of timesteps in the receptive field.
        """
        return self.eff_hist

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
