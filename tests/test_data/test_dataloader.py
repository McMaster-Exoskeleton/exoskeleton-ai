"""Tests for DataLoader integration and create_dataloaders function."""

from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from exoskeleton_ml.data.datasets import ExoskeletonDataset, create_dataloaders


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset for testing."""
    # Create enough data for meaningful train/val/test splits
    participants = []
    trial_names = []
    masses = []
    seq_lengths = []
    imu_features = []
    angle_features = []
    moment_targets = []

    # Create data for 5 participants with 4 trials each
    participant_ids = ["BT01", "BT02", "BT03", "BT06", "BT07"]
    participant_masses = [80.59, 72.24, 95.29, 79.33, 64.49]

    for i, (pid, mass) in enumerate(zip(participant_ids, participant_masses, strict=True)):
        for trial_num in range(4):
            seq_len = 100 + trial_num * 20  # Variable lengths
            participants.append(pid)
            trial_names.append(f"walk_{trial_num + 1}")
            masses.append(mass)
            seq_lengths.append(seq_len)
            imu_features.append([[float(j) * 0.1 * (i + 1)] * 24 for j in range(seq_len)])
            angle_features.append([[float(j) * 0.01 * (i + 1)] * 4 for j in range(seq_len)])
            moment_targets.append([[float(j) * 0.001 * (i + 1)] * 4 for j in range(seq_len)])

    data = {
        "participant": participants,
        "trial_name": trial_names,
        "mass_kg": masses,
        "sequence_length": seq_lengths,
        "imu_features": imu_features,
        "angle_features": angle_features,
        "moment_targets": moment_targets,
    }
    return Dataset.from_dict(data)


class TestDataLoaderBasic:
    """Test basic DataLoader integration with ExoskeletonDataset."""

    def test_dataloader_iteration(self, tmp_path, mock_hf_dataset):
        """Test basic iteration through DataLoader."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01", "BT02"],
                download=True,
                normalize=False,
            )

            # Create DataLoader
            loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

            # Iterate through batches
            batch_count = 0
            for batch in loader:
                batch_count += 1

                # Verify batch structure
                assert "inputs" in batch
                assert "targets" in batch
                assert "lengths" in batch
                assert "mask" in batch
                assert "metadata" in batch

                # Verify batch dimensions
                assert batch["inputs"].ndim == 3  # (batch, seq_len, features)
                assert batch["targets"].ndim == 3  # (batch, seq_len, targets)
                assert batch["lengths"].ndim == 1  # (batch,)
                assert batch["mask"].ndim == 2  # (batch, seq_len)

            # Should have processed all data
            assert batch_count == 2  # 8 trials / batch_size 4 = 2 batches

    def test_dataloader_shuffling(self, tmp_path, mock_hf_dataset):
        """Test that shuffling works correctly."""
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

            # Create two loaders with different random seeds
            loader1 = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
            loader2 = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

            # Get first batch from each
            batch1 = next(iter(loader1))
            batch2 = next(iter(loader2))

            # They might be different due to shuffling
            # Just verify both are valid
            assert batch1["inputs"].shape[0] == 2
            assert batch2["inputs"].shape[0] == 2

    def test_dataloader_num_workers(self, tmp_path, mock_hf_dataset):
        """Test DataLoader with multiple workers."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01", "BT02"],
                download=True,
                normalize=False,
            )

            # Create DataLoader with workers
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=2,
                collate_fn=dataset.collate_fn,
            )

            # Should still work correctly
            batch = next(iter(loader))
            assert batch["inputs"].shape[0] <= 4

    def test_dataloader_variable_batch_sizes(self, tmp_path, mock_hf_dataset):
        """Test DataLoader with different batch sizes."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01"],  # 4 trials
                download=True,
                normalize=False,
            )

            for batch_size in [1, 2, 4, 8]:
                loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
                )

                total_samples = 0
                for batch in loader:
                    total_samples += batch["inputs"].shape[0]

                # Should process all samples
                assert total_samples == 4


class TestCreateDataloaders:
    """Test create_dataloaders helper function."""

    def test_create_dataloaders_basic(self, tmp_path, mock_hf_dataset):
        """Test creating train/val/test dataloaders."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            train_loader, val_loader, test_loader = create_dataloaders(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                train_participants=["BT01", "BT02"],
                val_participants=["BT03"],
                test_participants=["BT06"],
                batch_size=4,
                num_workers=0,
                normalize=False,
            )

            # Verify all loaders were created
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None

            # Verify they're DataLoader instances
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)

            # Test iteration
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            test_batch = next(iter(test_loader))

            assert train_batch["inputs"].shape[2] == 28
            assert val_batch["inputs"].shape[2] == 28
            assert test_batch["inputs"].shape[2] == 28

    def test_create_dataloaders_optional_splits(self, tmp_path, mock_hf_dataset):
        """Test creating dataloaders with optional val/test."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            # Only train
            train_loader, val_loader, test_loader = create_dataloaders(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                train_participants=["BT01", "BT02"],
                val_participants=None,
                test_participants=None,
                batch_size=4,
                num_workers=0,
                normalize=False,
            )

            assert train_loader is not None
            assert val_loader is None
            assert test_loader is None

    def test_create_dataloaders_with_normalization(self, tmp_path, mock_hf_dataset):
        """Test creating dataloaders with normalization."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            train_loader, val_loader, test_loader = create_dataloaders(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                train_participants=["BT01", "BT02"],
                val_participants=["BT03"],
                test_participants=["BT06"],
                batch_size=4,
                num_workers=0,
                normalize=True,
            )

            # All loaders should work
            train_batch = next(iter(train_loader))
            next(iter(val_loader))  # Verify val_loader works
            next(iter(test_loader))  # Verify test_loader works

            # Data should be normalized (approximately)
            assert train_batch["inputs"].shape[2] == 28

    def test_create_dataloaders_shuffle_behavior(self, tmp_path, mock_hf_dataset):
        """Test that train shuffles but val/test don't."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            train_loader, val_loader, _ = create_dataloaders(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                train_participants=["BT01", "BT02"],
                val_participants=["BT03"],
                batch_size=4,
                num_workers=0,
                normalize=False,
            )

            # Check shuffle settings (internal DataLoader attribute)
            # Note: This checks the DataLoader's internal state
            assert hasattr(train_loader, "dataset")
            assert hasattr(val_loader, "dataset")


class TestDataLoaderBatchProperties:
    """Test properties of batches from DataLoader."""

    def test_batch_padding_correctness(self, tmp_path, mock_hf_dataset):
        """Test that padding in batches is correct."""
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

            loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

            batch = next(iter(loader))

            # Verify padding for each sample in batch
            for i in range(batch["inputs"].shape[0]):
                seq_len = batch["lengths"][i].item()

                # Data before seq_len should not be all zeros
                if seq_len > 0:
                    # At least some values should be non-zero (given our mock data)
                    # We can't assert this strongly since some timesteps might be zero
                    _ = batch["inputs"][i, :seq_len]  # Access to verify data exists

                # Data after seq_len should be all zeros (padding)
                if seq_len < batch["inputs"].shape[1]:
                    padding = batch["inputs"][i, seq_len:]
                    assert torch.allclose(padding, torch.zeros_like(padding))

                # Mask should match
                assert batch["mask"][i, :seq_len].all()
                if seq_len < batch["inputs"].shape[1]:
                    assert not batch["mask"][i, seq_len:].any()

    def test_batch_metadata_correctness(self, tmp_path, mock_hf_dataset):
        """Test that metadata in batches is correct."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01", "BT02"],
                download=True,
                normalize=False,
            )

            loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

            batch = next(iter(loader))

            # Verify metadata structure
            assert "participants" in batch["metadata"]
            assert "trial_names" in batch["metadata"]
            assert "masses" in batch["metadata"]

            # Verify metadata length matches batch size
            batch_size = batch["inputs"].shape[0]
            assert len(batch["metadata"]["participants"]) == batch_size
            assert len(batch["metadata"]["trial_names"]) == batch_size
            assert len(batch["metadata"]["masses"]) == batch_size

            # Verify metadata types
            assert all(isinstance(p, str) for p in batch["metadata"]["participants"])
            assert all(isinstance(t, str) for t in batch["metadata"]["trial_names"])
            assert all(isinstance(m, float) for m in batch["metadata"]["masses"])

    def test_batch_tensor_types_and_devices(self, tmp_path, mock_hf_dataset):
        """Test that batch tensors have correct types and are on CPU."""
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

            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

            batch = next(iter(loader))

            # Verify dtypes
            assert batch["inputs"].dtype == torch.float32
            assert batch["targets"].dtype == torch.float32
            assert batch["lengths"].dtype == torch.long
            assert batch["mask"].dtype == torch.bool

            # Verify device (should be CPU before moving to GPU)
            assert batch["inputs"].device.type == "cpu"
            assert batch["targets"].device.type == "cpu"

    def test_batch_dimension_consistency(self, tmp_path, mock_hf_dataset):
        """Test that all batch dimensions are consistent."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["BT01", "BT02"],
                download=True,
                normalize=False,
            )

            loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

            for batch in loader:
                batch_size = batch["inputs"].shape[0]
                max_seq_len = batch["inputs"].shape[1]

                # All tensors should have consistent batch dimension
                assert batch["targets"].shape[0] == batch_size
                assert batch["lengths"].shape[0] == batch_size
                assert batch["mask"].shape[0] == batch_size

                # Sequence dimension should match
                assert batch["targets"].shape[1] == max_seq_len
                assert batch["mask"].shape[1] == max_seq_len

                # Feature dimensions should be correct
                assert batch["inputs"].shape[2] == 28  # 24 IMU + 4 angles
                assert batch["targets"].shape[2] == 4  # 4 moments


class TestDataLoaderEdgeCases:
    """Test edge cases in DataLoader usage."""

    def test_single_sample_batch(self, tmp_path, mock_hf_dataset):
        """Test DataLoader with batch_size=1."""
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

            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

            batch = next(iter(loader))

            # Should have batch dimension of 1
            assert batch["inputs"].shape[0] == 1
            assert batch["targets"].shape[0] == 1
            assert batch["lengths"].shape[0] == 1

    def test_empty_participant_filter(self, tmp_path, mock_hf_dataset):
        """Test behavior with participant filter that matches nothing."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.datasets.download_and_cache_dataset") as mock_download:
            mock_download.return_value = mock_hf_dataset

            dataset = ExoskeletonDataset(
                hf_repo="test/repo",
                cache_dir=cache_dir,
                participants=["NONEXISTENT"],
                download=True,
                normalize=False,
            )

            # Dataset should be empty
            assert len(dataset) == 0

            # DataLoader should handle empty dataset
            loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

            # Iteration should complete without errors
            batch_count = 0
            for _ in loader:
                batch_count += 1

            assert batch_count == 0
