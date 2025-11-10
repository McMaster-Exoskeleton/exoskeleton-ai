"""Evaluation metrics for model assessment."""

import numpy as np
import torch


def compute_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    average: str = "macro",
) -> dict[str, float]:
    pass
