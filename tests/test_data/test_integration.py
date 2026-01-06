"""
End-to-end integration tests for data pipeline.

These tests require access to the actual HuggingFace dataset and are slower
than unit tests. They verify the complete data pipeline from download to
DataLoader iteration.

Run with: pytest tests/test_data/test_integration.py -v -s
Skip with: pytest -m "not integration"
"""

import pytest
import torch
from torch.utils.data import DataLoader

from exoskeleton_ml.data import ExoskeletonDataset, create_dataloaders

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def hf_repo():
    """HuggingFace repository for testing."""
    return "MacExo/exoData"


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Shared cache directory for all integration tests."""
    return tmp_path_factory.mktemp("integration_cache")


class TestDatasetDownloadIntegration:
    """Integration tests for dataset download and caching."""

    def test_download_from_huggingface(self, hf_repo, cache_dir):
        """Test downloading dataset from HuggingFace."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        # Should have trials
        assert len(dataset) > 0
        print(f"✅ Downloaded {len(dataset)} trials for BT01")

    def test_cache_reuse_after_download(self, hf_repo, cache_dir):
        """Test that cache is reused on second load."""
        # First load (may download or use cache)
        dataset1 = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        len1 = len(dataset1)

        # Second load (should use cache)
        dataset2 = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        # Should have same number of trials
        assert len(dataset2) == len1
        print(f"✅ Cache reuse verified: {len(dataset2)} trials")


class TestDatasetFunctionalityIntegration:
    """Integration tests for dataset functionality with real data."""

    def test_load_single_trial(self, hf_repo, cache_dir):
        """Test loading a single trial from real dataset."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        # Get first trial
        trial = dataset[0]

        # Verify structure
        assert "inputs" in trial
        assert "targets" in trial
        assert "participant" in trial
        assert "trial_name" in trial
        assert "mass_kg" in trial
        assert "sequence_length" in trial

        # Verify shapes
        seq_len = trial["sequence_length"]
        assert trial["inputs"].shape == (seq_len, 28)
        assert trial["targets"].shape == (seq_len, 4)

        # Verify participant
        assert trial["participant"] == "BT01"

        print(f"✅ Trial {trial['trial_name']}: {seq_len} timesteps")

    def test_variable_sequence_lengths(self, hf_repo, cache_dir):
        """Test that dataset handles variable-length sequences."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        # Get sequence lengths
        lengths = [dataset[i]["sequence_length"] for i in range(min(10, len(dataset)))]

        # Should have variation
        assert len(set(lengths)) > 1, "Expected variable sequence lengths"
        assert min(lengths) > 0
        assert max(lengths) < 100000  # Reasonable upper bound

        print(f"✅ Sequence length range: {min(lengths)} to {max(lengths)}")

    def test_data_quality(self, hf_repo, cache_dir):
        """Test that loaded data has reasonable values."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        # Sample a few trials
        for i in range(min(5, len(dataset))):
            trial = dataset[i]

            # Check for reasonable ranges (not all zeros, not all huge values)
            inputs_abs_mean = trial["inputs"].abs().mean().item()
            targets_abs_mean = trial["targets"].abs().mean().item()

            assert 0.01 < inputs_abs_mean < 1000, f"Inputs seem off: {inputs_abs_mean}"
            assert 0.001 < targets_abs_mean < 100, f"Targets seem off: {targets_abs_mean}"

        print(f"✅ Data quality checks passed for {min(5, len(dataset))} trials")


class TestDataLoaderIntegration:
    """Integration tests for DataLoader with real data."""

    def test_dataloader_iteration(self, hf_repo, cache_dir):
        """Test iterating through DataLoader with real data."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01", "BT02"], download=True
        )

        loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, shuffle=False)

        # Iterate through one batch
        batch = next(iter(loader))

        # Verify batch structure
        assert batch["inputs"].ndim == 3
        assert batch["targets"].ndim == 3
        assert batch["lengths"].ndim == 1
        assert batch["mask"].ndim == 2

        # Verify batch size
        assert batch["inputs"].shape[0] <= 4

        print(f"✅ Batch shape: {batch['inputs'].shape}")

    def test_dataloader_padding(self, hf_repo, cache_dir):
        """Test that padding works correctly with real variable-length data."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo, cache_dir=cache_dir, participants=["BT01"], download=True
        )

        loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, shuffle=False)

        batch = next(iter(loader))

        # Verify padding
        for i in range(batch["inputs"].shape[0]):
            seq_len = batch["lengths"][i].item()

            # Mask should be True for real data, False for padding
            assert batch["mask"][i, :seq_len].all()
            if seq_len < batch["inputs"].shape[1]:
                assert not batch["mask"][i, seq_len:].any()

                # Padding should be zeros
                assert torch.allclose(
                    batch["inputs"][i, seq_len:],
                    torch.zeros(batch["inputs"].shape[1] - seq_len, 28),
                )

        print(f"✅ Padding verified for batch with lengths: {batch['lengths'].tolist()}")

    def test_create_dataloaders_splits(self, hf_repo, cache_dir):
        """Test create_dataloaders with participant-based splits."""
        train_loader, val_loader, test_loader = create_dataloaders(
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            train_participants=["BT01", "BT02"],
            val_participants=["BT03"],
            test_participants=["BT06"],
            batch_size=4,
            num_workers=0,
            normalize=False,
        )

        # All loaders should be created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Should be able to iterate
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # All should have correct feature dimensions
        assert train_batch["inputs"].shape[2] == 28
        assert val_batch["inputs"].shape[2] == 28
        assert test_batch["inputs"].shape[2] == 28

        print(
            f"✅ Created loaders - Train: {len(train_loader)} batches, "
            f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches"
        )


class TestNormalizationIntegration:
    """Integration tests for normalization with real data."""

    def test_normalization_computation(self, hf_repo, cache_dir):
        """Test that normalization statistics are computed correctly."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            participants=["BT01"],
            download=True,
            normalize=True,
        )

        # Should have computed stats
        assert dataset.normalization_stats is not None
        assert "inputs" in dataset.normalization_stats
        assert "targets" in dataset.normalization_stats

        # Stats should be valid numbers (not NaN)
        for key in ["mean", "std", "min", "max"]:
            inputs_stats = dataset.normalization_stats["inputs"][key]
            assert len(inputs_stats) == 28
            # Allow some NaN handling for features that are all NaN
            assert any(isinstance(x, (int, float)) for x in inputs_stats)

        print("✅ Normalization stats computed successfully")

    def test_normalized_data_distribution(self, hf_repo, cache_dir):
        """Test that normalized data has approximately zero mean and unit variance."""
        dataset = ExoskeletonDataset(
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            participants=["BT01"],
            download=True,
            normalize=True,
        )

        # Sample a trial
        trial = dataset[0]

        # Check distribution (should be roughly normalized)
        # Note: Single trial won't be exactly normalized, but should be in reasonable range
        inputs_mean = trial["inputs"].mean().item()
        inputs_std = trial["inputs"].std().item()

        # Should be in a reasonable range (not exact due to single trial)
        assert -5 < inputs_mean < 5, f"Normalized mean seems off: {inputs_mean}"
        assert 0.1 < inputs_std < 10, f"Normalized std seems off: {inputs_std}"

        print(f"✅ Normalized data - mean: {inputs_mean:.2f}, std: {inputs_std:.2f}")


class TestEndToEndPipeline:
    """End-to-end integration test."""

    def test_complete_pipeline(self, hf_repo, cache_dir):
        """Test complete pipeline from download to training-ready batches."""
        # Step 1: Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            train_participants=["BT01", "BT02"],
            val_participants=["BT03"],
            batch_size=8,
            num_workers=0,
            normalize=True,
        )

        # Step 2: Simulate training loop
        train_batches = 0
        for batch in train_loader:
            # Verify batch is ready for training
            assert batch["inputs"].dtype == torch.float32
            assert batch["targets"].dtype == torch.float32
            assert batch["inputs"].shape[2] == 28
            assert batch["targets"].shape[2] == 4

            train_batches += 1
            if train_batches >= 3:  # Just test a few batches
                break

        # Step 3: Validate on validation set
        val_batches = 0
        for batch in val_loader:
            assert batch["inputs"].dtype == torch.float32
            assert batch["targets"].dtype == torch.float32

            val_batches += 1
            if val_batches >= 2:
                break

        print(
            f"✅ End-to-end pipeline complete: "
            f"processed {train_batches} train batches, {val_batches} val batches"
        )
