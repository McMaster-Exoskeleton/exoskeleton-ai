"""Model factory for creating models from configuration."""

import torch.nn as nn
from omegaconf import DictConfig

from .baseline import BaselineModel
from .tcn import TCN


def create_model(config: DictConfig) -> nn.Module:
    """Create a model based on configuration.

    Args:
        config: Model configuration from configs/model/*.yaml

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If model type is not recognized.
    """
    model_type = config.type

    if model_type == "baseline":
        return BaselineModel(
            _input_size=config.get("input_size", 10),
            _hidden_size=config.get("hidden_size", 128),
            _num_classes=config.get("num_classes", 5),
            _num_layers=config.get("num_layers", 2),
            _dropout=config.get("dropout", 0.2),
        )

    elif model_type == "tcn":
        # Calculate effective history
        num_levels = len(config.architecture.num_channels)
        kernel_size = config.architecture.kernel_size
        eff_hist = (kernel_size - 1) * (2**num_levels - 1) + 1

        return TCN(
            input_size=config.architecture.input_size,
            output_size=config.architecture.output_size,
            num_channels=config.architecture.num_channels,
            kernel_size=kernel_size,
            dropout=config.architecture.dropout,
            eff_hist=eff_hist,
            spatial_dropout=config.architecture.spatial_dropout,
            activation=config.architecture.activation,
            norm=config.architecture.norm,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
