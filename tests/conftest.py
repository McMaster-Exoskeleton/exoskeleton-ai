"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample data for testing.

    Returns:
        Tuple of (features, labels) tensors.
    """
    batch_size = 8
    seq_len = 100
    n_features = 10
    n_classes = 5

    features = torch.randn(batch_size, seq_len, n_features)
    labels = torch.randint(0, n_classes, (batch_size,))

    return features, labels


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model checkpoints."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
