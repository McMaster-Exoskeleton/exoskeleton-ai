"""Tests for data download and caching functionality."""

import json
from unittest.mock import patch

import pytest
from datasets import Dataset

from exoskeleton_ml.data.download import (
    clear_cache,
    download_and_cache_dataset,
    get_cache_info,
)


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset for testing."""
    # Create sample data
    data = {
        "participant": ["BT01", "BT01", "BT02"],
        "trial_name": ["walk_1", "walk_2", "walk_1"],
        "mass_kg": [80.59, 80.59, 72.24],
        "sequence_length": [100, 150, 120],
        "imu_features": [
            [[0.0] * 24 for _ in range(100)],
            [[0.1] * 24 for _ in range(150)],
            [[0.2] * 24 for _ in range(120)],
        ],
        "angle_features": [
            [[0.0] * 4 for _ in range(100)],
            [[0.1] * 4 for _ in range(150)],
            [[0.2] * 4 for _ in range(120)],
        ],
        "moment_targets": [
            [[0.0] * 4 for _ in range(100)],
            [[0.1] * 4 for _ in range(150)],
            [[0.2] * 4 for _ in range(120)],
        ],
    }
    return Dataset.from_dict(data)


class TestDownloadAndCache:
    """Test suite for download_and_cache_dataset function."""

    def test_download_new_dataset(self, tmp_path, mock_hf_dataset):
        """Test downloading a dataset for the first time."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset

            dataset = download_and_cache_dataset(
                hf_repo="test/repo", cache_dir=cache_dir, verify_integrity=True
            )

            # Verify dataset was returned
            assert len(dataset) == 3
            assert dataset[0]["participant"] == "BT01"

            # Verify cache directory was created
            assert cache_dir.exists()
            assert (cache_dir / "metadata.json").exists()
            assert (cache_dir / "hf_cache").exists()

            # Verify metadata
            with open(cache_dir / "metadata.json") as f:
                metadata = json.load(f)
            assert metadata["hf_repo"] == "test/repo"
            assert metadata["total_trials"] == 3
            assert len(metadata["participants"]) == 2
            assert "BT01" in metadata["participants"]
            assert "BT02" in metadata["participants"]

    def test_load_from_cache(self, tmp_path, mock_hf_dataset):
        """Test loading dataset from existing cache."""
        cache_dir = tmp_path / "cache"

        # First download
        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset
            download_and_cache_dataset(hf_repo="test/repo", cache_dir=cache_dir)

        # Second call should load from cache
        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            dataset = download_and_cache_dataset(hf_repo="test/repo", cache_dir=cache_dir)

            # Should not call load_dataset again
            mock_load.assert_not_called()

            # Should still return correct dataset
            assert len(dataset) == 3

    def test_force_redownload(self, tmp_path, mock_hf_dataset):
        """Test force re-downloading even when cache exists."""
        cache_dir = tmp_path / "cache"

        # First download
        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset
            download_and_cache_dataset(hf_repo="test/repo", cache_dir=cache_dir)

        # Force re-download
        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset
            dataset = download_and_cache_dataset(
                hf_repo="test/repo", cache_dir=cache_dir, force_redownload=True
            )

            # Should call load_dataset again
            mock_load.assert_called_once()
            assert len(dataset) == 3

    def test_download_failure(self, tmp_path):
        """Test handling of download failures."""
        cache_dir = tmp_path / "cache"

        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Network error")

            with pytest.raises(FileNotFoundError) as exc_info:
                download_and_cache_dataset(hf_repo="test/repo", cache_dir=cache_dir)

            assert "Failed to download dataset" in str(exc_info.value)

    def test_integrity_verification_missing_fields(self, tmp_path):
        """Test integrity check fails with missing required fields."""
        cache_dir = tmp_path / "cache"

        # Create dataset with missing field
        bad_data = {
            "participant": ["BT01"],
            "trial_name": ["walk_1"],
            # Missing other required fields
        }
        bad_dataset = Dataset.from_dict(bad_data)

        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = bad_dataset

            with pytest.raises(ValueError) as exc_info:
                download_and_cache_dataset(
                    hf_repo="test/repo", cache_dir=cache_dir, verify_integrity=True
                )

            assert "missing required fields" in str(exc_info.value)

    def test_integrity_verification_wrong_dimensions(self, tmp_path):
        """Test integrity check fails with wrong feature dimensions."""
        cache_dir = tmp_path / "cache"

        # Create dataset with wrong dimensions
        bad_data = {
            "participant": ["BT01"],
            "trial_name": ["walk_1"],
            "mass_kg": [80.59],
            "sequence_length": [100],
            "imu_features": [[[0.0] * 20 for _ in range(100)]],  # Wrong: should be 24
            "angle_features": [[[0.0] * 4 for _ in range(100)]],
            "moment_targets": [[[0.0] * 4 for _ in range(100)]],
        }
        bad_dataset = Dataset.from_dict(bad_data)

        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = bad_dataset

            with pytest.raises(ValueError) as exc_info:
                download_and_cache_dataset(
                    hf_repo="test/repo", cache_dir=cache_dir, verify_integrity=True
                )

            assert "should have 24 dimensions" in str(exc_info.value)

    def test_integrity_verification_mismatched_lengths(self, tmp_path):
        """Test integrity check fails with mismatched sequence lengths."""
        cache_dir = tmp_path / "cache"

        # Create dataset with mismatched lengths
        bad_data = {
            "participant": ["BT01"],
            "trial_name": ["walk_1"],
            "mass_kg": [80.59],
            "sequence_length": [100],
            "imu_features": [[[0.0] * 24 for _ in range(50)]],  # Wrong length
            "angle_features": [[[0.0] * 4 for _ in range(100)]],
            "moment_targets": [[[0.0] * 4 for _ in range(100)]],
        }
        bad_dataset = Dataset.from_dict(bad_data)

        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = bad_dataset

            with pytest.raises(ValueError) as exc_info:
                download_and_cache_dataset(
                    hf_repo="test/repo", cache_dir=cache_dir, verify_integrity=True
                )

            assert "doesn't match sequence_length" in str(exc_info.value)


class TestCacheManagement:
    """Test suite for cache management utilities."""

    def test_get_cache_info_nonexistent(self, tmp_path):
        """Test getting info for non-existent cache."""
        cache_dir = tmp_path / "nonexistent"

        info = get_cache_info(cache_dir)

        assert info["exists"] is False
        assert info["size_gb"] == 0
        assert info["num_files"] == 0
        assert info["metadata"] is None

    def test_get_cache_info_existing(self, tmp_path, mock_hf_dataset):
        """Test getting info for existing cache."""
        cache_dir = tmp_path / "cache"

        # Create cache
        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset
            download_and_cache_dataset(hf_repo="test/repo", cache_dir=cache_dir)

        # Get info
        info = get_cache_info(cache_dir)

        assert info["exists"] is True
        assert info["size_gb"] > 0
        assert info["num_files"] > 0
        assert info["metadata"] is not None
        assert info["metadata"]["hf_repo"] == "test/repo"
        assert info["path"] == str(cache_dir)

    def test_clear_cache_nonexistent(self, tmp_path, capsys):
        """Test clearing non-existent cache."""
        cache_dir = tmp_path / "nonexistent"

        clear_cache(cache_dir)

        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_clear_cache_existing(self, tmp_path, mock_hf_dataset, capsys):
        """Test clearing existing cache."""
        cache_dir = tmp_path / "cache"

        # Create cache
        with patch("exoskeleton_ml.data.download.load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset
            download_and_cache_dataset(hf_repo="test/repo", cache_dir=cache_dir)

        # Verify cache exists
        assert (cache_dir / "metadata.json").exists()

        # Clear cache
        clear_cache(cache_dir)

        # Verify cache is cleared
        assert cache_dir.exists()  # Directory remains
        assert not (cache_dir / "metadata.json").exists()  # But files are gone

        captured = capsys.readouterr()
        assert "cleared successfully" in captured.out
