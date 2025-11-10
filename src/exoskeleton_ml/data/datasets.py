"""Dataset classes for exoskeleton data."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ExoskeletonDataset(Dataset):
    """Dataset for exoskeleton movement data.

    This is a template - implement based on your data format.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to dataset directory.
            split: Data split ('train', 'val', or 'test').
            transform: Optional transform to apply to samples.
        """
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (sample, label).
        """
        ...
