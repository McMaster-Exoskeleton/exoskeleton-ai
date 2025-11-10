"""Training loop implementation."""

from pathlib import Path

import torch
import torch.nn as nn

from exoskeleton_ml.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Dummy Trainer so Far"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        pass
