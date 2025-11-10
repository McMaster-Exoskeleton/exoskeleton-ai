"""Model factory for creating models from configuration."""

import torch.nn as nn
from omegaconf import DictConfig


def create_model(config: DictConfig) -> nn.Module:
    """Create a model based on configuration.

    Args:
        config: Model configuration.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model type is not recognized.
    """
    ...
