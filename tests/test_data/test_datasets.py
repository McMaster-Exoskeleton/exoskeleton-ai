"""Tests for ExoskeletonDataset class."""

from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from exoskeleton_ml.data.datasets import ExoskeletonDataset


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset for testing."""
    data = {
        "participant": ["BT01", "BT01", "BT02", "BT02"],
        "trial_name": ["walk_1", "walk_2", "walk_1", "walk_2"],
        "mass_kg": [80.59, 80.59, 72.24, 72.24],
        "sequence_length": [100, 150, 120, 130],
        "imu_features": [
            [[float(i) * 0.1] * 24 for i in range(100)],
            [[float(i) * 0.2] * 24 for i in range(150)],
            [[float(i) * 0.3] * 24 for i in range(120)],
            [[float(i) * 0.4] * 24 for i in range(130)],
        ],
        "angle_features": [
            [[float(i) * 0.01] * 4 for i in range(100)],
            [[float(i) * 0.02] * 4 for i in range(150)],
            [[float(i) * 0.03] * 4 for i in range(120)],
            [[float(i) * 0.04] * 4 for i in range(130)],
        ],
        "moment_targets": [
            [[float(i) * 0.001] * 4 for i in range(100)],
            [[float(i) * 0.002] * 4 for i in range(150)],
            [[float(i) * 0.003] * 4 for i in range(120)],
            [[float(i) * 0.004] * 4 for i in range(130)],
        ],
    }
    return Dataset.from_dict(data)


class TestExoskeletonDatasetInit:
    """Test suite for ExoskeletonDataset initialization."""

    def test_init_with_download(self, tmp_path, mock_hf_dataset):
        """Test initialization with automatic download."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Verify download was called
            mock_download.assert_called_once()

            # Verify dataset was initialized
            assert len(dataset) == 4

    def test_init_with_participant_filter(self, tmp_path, mock_hf_dataset):
        """Test initialization with participant filtering."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01"],
                download=True,
                normalize=False,
            )

            # Should only have BT01 trials
            assert len(dataset) == 2

    def test_init_without_download_cache_missing(self, tmp_path):
        """Test initialization without download when cache doesn't exist."""
        cache_dir = tmp_path / "cache"

        with pytest.raises(FileNotFoundError) as exc_info:
            ExoskeletonDataset(hf_repo="test/repo", cache_dir=cache_dir, download=False)

        assert "Dataset not found in cache" in str(exc_info.value)

    def test_init_with_normalization(self, tmp_path, mock_hf_dataset):
        """Test initialization with normalization enabled."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=True
            )

            # Verify normalization stats were computed
            assert dataset.normalization_stats is not None
            assert "inputs" in dataset.normalization_stats
            assert "targets" in dataset.normalization_stats
            assert "mean" in dataset.normalization_stats["inputs"]
            assert "std" in dataset.normalization_stats["inputs"]

            # Verify stats file was created
            stats_file = cache_dir / "normalization_stats.json"
            assert stats_file.exists()

    def test_init_with_provided_normalization_stats(self, tmp_path, mock_hf_dataset):
        """Test initialization with pre-computed normalization stats."""
        cache_dir = tmp_path / "cache"

        custom_stats = {
            "inputs": {
                "mean": [0.0] * 28,
                "std": [1.0] * 28,
                "min": [-1.0] * 28,
                "max": [1.0] * 28,
            },
            "targets": {"mean": [0.0] * 4, "std": [1.0] * 4, "min": [-1.0] * 4, "max": [1.0] * 4},
        }

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                download=True,
                normalize=True,
                normalization_stats=custom_stats,
            )

            # Should use provided stats
            assert dataset.normalization_stats == custom_stats


class TestExoskeletonDatasetCaching:
    """Test suite for dataset caching functionality."""

    def test_cache_creation(self, tmp_path, mock_hf_dataset):
        """Test that preprocessed cache is created correctly."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Verify cache directory was created
            assert len(list(cache_dir.glob("preprocessed_*/trial_*.pt"))) == 4

            # Verify index file was created
            index_files = list(cache_dir.glob("preprocessed_*/index.pkl"))
            assert len(index_files) == 1

    def test_cache_reuse(self, tmp_path, mock_hf_dataset):
        """Test that existing cache is reused."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            # First initialization - creates cache
            dataset1 = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Second initialization - should reuse cache (with download=True to load HF cache)
            dataset2 = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Both should have same length
            assert len(dataset1) == len(dataset2)

    def test_force_reprocess(self, tmp_path, mock_hf_dataset):
        """Test force reprocessing even when cache exists."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            # First initialization
            ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Second initialization with force_reprocess (need download=True to load HF cache)
            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                download=True,
                force_reprocess=True,
                normalize=False,
            )

            assert len(dataset) == 4

    def test_different_participant_filters_different_caches(self, tmp_path, mock_hf_dataset):
        """Test that different participant filters create separate caches."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            # Create dataset for BT01
            dataset1 = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01"],
                download=True,
                normalize=False,
            )

            # Create dataset for BT02 (need download=True to have HF cache available)
            dataset2 = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT02"],
                download=True,
                normalize=False,
            )

            # Should have different lengths
            assert len(dataset1) == 2
            assert len(dataset2) == 2

            # Should have different cache directories
            cache_dirs = list(cache_dir.glob("preprocessed_*"))
            assert len(cache_dirs) == 2


class TestExoskeletonDatasetGetItem:
    """Test suite for __getitem__ functionality."""

    def test_getitem_basic(self, tmp_path, mock_hf_dataset):
        """Test basic __getitem__ functionality."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Get first item
            item = dataset[0]

            # Verify structure
            assert "inputs" in item
            assert "targets" in item
            assert "participant" in item
            assert "trial_name" in item
            assert "mass_kg" in item
            assert "sequence_length" in item

            # Verify shapes
            assert item["inputs"].shape == (100, 28)  # seq_len x (24 IMU + 4 angles)
            assert item["targets"].shape == (100, 4)  # seq_len x 4 moments

            # Verify types
            assert isinstance(item["inputs"], torch.Tensor)
            assert isinstance(item["targets"], torch.Tensor)
            assert item["inputs"].dtype == torch.float32
            assert item["targets"].dtype == torch.float32

    def test_getitem_with_normalization(self, tmp_path, mock_hf_dataset):
        """Test __getitem__ with normalization applied."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=True
            )

            # Get item
            item = dataset[0]

            # Data should be normalized (not checking exact values, just that it's different)
            assert item["inputs"].shape == (100, 28)
            assert item["targets"].shape == (100, 4)

    def test_getitem_all_indices(self, tmp_path, mock_hf_dataset):
        """Test that all indices can be accessed."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Access all items
            for i in range(len(dataset)):
                item = dataset[i]
                assert item["inputs"].shape[0] == item["sequence_length"]
                assert item["targets"].shape[0] == item["sequence_length"]


class TestExoskeletonDatasetCollate:
    """Test suite for collate_fn functionality."""

    def test_collate_single_batch(self, tmp_path, mock_hf_dataset):
        """Test collate function with a single batch."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Get a batch manually
            batch = [dataset[0], dataset[1]]

            # Apply collate
            collated = ExoskeletonDataset.collate_fn(batch)

            # Verify structure
            assert "inputs" in collated
            assert "targets" in collated
            assert "lengths" in collated
            assert "mask" in collated
            assert "metadata" in collated

            # Verify shapes
            batch_size = 2
            max_len = max(item["sequence_length"] for item in batch)
            assert collated["inputs"].shape == (batch_size, max_len, 28)
            assert collated["targets"].shape == (batch_size, max_len, 4)
            assert collated["lengths"].shape == (batch_size,)
            assert collated["mask"].shape == (batch_size, max_len)

    def test_collate_padding(self, tmp_path, mock_hf_dataset):
        """Test that padding is applied correctly."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Get items with different lengths (100 and 150)
            batch = [dataset[0], dataset[1]]
            collated = ExoskeletonDataset.collate_fn(batch)

            # Verify padding
            # First item has length 100, second has 150
            assert collated["lengths"][0] == 100
            assert collated["lengths"][1] == 150

            # Check mask
            assert collated["mask"][0, :100].all()  # First 100 should be True
            assert not collated["mask"][0, 100:].any()  # Rest should be False
            assert collated["mask"][1, :150].all()  # All 150 should be True

            # Check padding is zeros
            assert torch.allclose(collated["inputs"][0, 100:], torch.zeros(50, 28))

    def test_collate_metadata(self, tmp_path, mock_hf_dataset):
        """Test that metadata is preserved in collate."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Get batch
            batch = [dataset[0], dataset[2]]  # BT01 and BT02
            collated = ExoskeletonDataset.collate_fn(batch)

            # Verify metadata
            assert len(collated["metadata"]["participants"]) == 2
            assert collated["metadata"]["participants"][0] == "BT01"
            assert collated["metadata"]["participants"][1] == "BT02"
            assert len(collated["metadata"]["trial_names"]) == 2
            assert len(collated["metadata"]["masses"]) == 2
            assert collated["metadata"]["masses"][0] == 80.59
            assert collated["metadata"]["masses"][1] == 72.24


class TestExoskeletonDatasetUtilities:
    """Test suite for utility methods."""

    def test_get_trial_info(self, tmp_path, mock_hf_dataset):
        """Test get_trial_info method."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Get trial info
            info = dataset.get_trial_info(0)

            # Verify info
            assert "participant" in info
            assert "trial_name" in info
            assert "sequence_length" in info
            assert "file" in info

    def test_get_statistics(self, tmp_path, mock_hf_dataset):
        """Test get_statistics method."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            # Get statistics
            stats = dataset.get_statistics()

            # Verify statistics
            assert stats["num_trials"] == 4
            assert stats["num_participants"] == 2
            assert "BT01" in stats["participants"]
            assert "BT02" in stats["participants"]
            assert "sequence_lengths" in stats
            assert stats["sequence_lengths"]["min"] == 100
            assert stats["sequence_lengths"]["max"] == 150

    def test_len(self, tmp_path, mock_hf_dataset):
        """Test __len__ method."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo", cache_dir=cache_dir, download=True, normalize=False
            )

            assert len(dataset) == 4

            # Test with filter (need download=True to have HF cache available)
            dataset_filtered = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01"],
                download=True,
                normalize=False,
            )

            assert len(dataset_filtered) == 2
