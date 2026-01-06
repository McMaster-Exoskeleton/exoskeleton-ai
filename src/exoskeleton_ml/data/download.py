"""Data download utilities for exoskeleton datasets."""

import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk


def download_and_cache_dataset(
    hf_repo: str,
    cache_dir: str | Path,
    force_redownload: bool = False,
    verify_integrity: bool = True,
) -> Dataset:
    """
    Download dataset from HuggingFace and cache locally.

    This function handles:
    1. Checking if data is already cached
    2. Downloading from HuggingFace if needed
    3. Caching the dataset locally
    4. Validating dataset integrity
    5. Saving metadata

    Args:
        hf_repo: HuggingFace repository ID (e.g., "MacExo/exoData")
        cache_dir: Local cache directory
        force_redownload: Force re-download even if cached
        verify_integrity: Verify dataset integrity after download

    Returns:
        HuggingFace Dataset object

    Raises:
        FileNotFoundError: If dataset cannot be downloaded
        ValueError: If dataset fails integrity checks

    Example:
        >>> dataset = download_and_cache_dataset(
        ...     "MacExo/exoData",
        ...     "data/processed/phase1"
        ... )
        >>> print(f"Loaded {len(dataset)} trials")
    """
    cache_dir = Path(cache_dir)
    metadata_file = cache_dir / "metadata.json"
    hf_cache_dir = cache_dir / "hf_cache"

    # Check if already cached
    if metadata_file.exists() and not force_redownload:
        print(f"✅ Dataset already cached at {cache_dir}")

        # Load metadata
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            print(f"   Repository: {metadata['hf_repo']}")
            print(f"   Participants: {metadata['participants']}")
            print(f"   Total trials: {metadata['total_trials']}")
            print(f"   Downloaded: {metadata['download_date']}")

            # Load cached dataset
            if hf_cache_dir.exists():
                dataset = load_from_disk(str(hf_cache_dir))
                print(f"✅ Loaded from cache: {len(dataset)} trials")
                return dataset
            else:
                print("⚠️  Cache metadata exists but dataset files missing. Re-downloading...")

        except Exception as e:
            print(f"⚠️  Failed to load cached dataset: {e}")
            print("   Re-downloading from HuggingFace...")

    # Download from HuggingFace
    print(f"⬇️  Downloading dataset from {hf_repo}...")
    print("   This may take several minutes on first download...")

    try:
        # Load dataset (HuggingFace will cache in ~/.cache/huggingface/datasets/)
        dataset = load_dataset(hf_repo, split="train")
        print(f"✅ Downloaded {len(dataset)} trials from HuggingFace")

    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download dataset from {hf_repo}. "
            f"Error: {e}\n"
            f"Please check:\n"
            f"  1. Repository name is correct\n"
            f"  2. You have internet connection\n"
            f"  3. Repository is public or you're authenticated (huggingface-cli login)"
        ) from e

    # Verify integrity
    if verify_integrity:
        print("🔍 Verifying dataset integrity...")
        _verify_dataset_integrity(dataset)
        print("✅ Dataset integrity verified")

    # Save to local cache
    print(f"💾 Caching dataset to {cache_dir}...")
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(hf_cache_dir))

    # Save metadata
    participants = sorted(dataset.unique("participant"))
    metadata = {
        "hf_repo": hf_repo,
        "participants": participants,
        "total_trials": len(dataset),
        "download_date": datetime.now().isoformat(),
        "num_participants": len(participants),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Cached {len(dataset)} trials from {len(participants)} participants")
    print(f"   Cache location: {cache_dir}")

    # Print summary
    _print_dataset_summary(dataset)

    return dataset


def _verify_dataset_integrity(dataset: Dataset) -> None:
    """
    Verify dataset has expected structure and valid data.

    Args:
        dataset: HuggingFace Dataset to verify

    Raises:
        ValueError: If dataset fails integrity checks
    """
    required_fields = [
        "participant",
        "trial_name",
        "mass_kg",
        "sequence_length",
        "imu_features",
        "angle_features",
        "moment_targets",
    ]

    # Check required fields exist
    missing_fields = [field for field in required_fields if field not in dataset.column_names]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")

    # Sample first trial for validation
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    sample = dataset[0]

    # Check data types
    if not isinstance(sample["participant"], str):
        raise ValueError("Field 'participant' should be string")

    if not isinstance(sample["trial_name"], str):
        raise ValueError("Field 'trial_name' should be string")

    if not isinstance(sample["sequence_length"], int):
        raise ValueError("Field 'sequence_length' should be int")

    # Check feature dimensions
    seq_len = sample["sequence_length"]

    if len(sample["imu_features"]) != seq_len:
        raise ValueError(
            f"IMU features length ({len(sample['imu_features'])}) "
            f"doesn't match sequence_length ({seq_len})"
        )

    if len(sample["angle_features"]) != seq_len:
        raise ValueError(
            f"Angle features length ({len(sample['angle_features'])}) "
            f"doesn't match sequence_length ({seq_len})"
        )

    if len(sample["moment_targets"]) != seq_len:
        raise ValueError(
            f"Moment targets length ({len(sample['moment_targets'])}) "
            f"doesn't match sequence_length ({seq_len})"
        )

    # Check feature vector dimensions
    if len(sample["imu_features"][0]) != 24:
        raise ValueError(
            f"IMU features should have 24 dimensions, got {len(sample['imu_features'][0])}"
        )

    if len(sample["angle_features"][0]) != 4:
        raise ValueError(
            f"Angle features should have 4 dimensions, got {len(sample['angle_features'][0])}"
        )

    if len(sample["moment_targets"][0]) != 4:
        raise ValueError(
            f"Moment targets should have 4 dimensions, got {len(sample['moment_targets'][0])}"
        )


def _print_dataset_summary(dataset: Dataset) -> None:
    """Print summary statistics of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    participants = sorted(dataset.unique("participant"))
    print(f"Total trials: {len(dataset)}")
    print(f"Participants: {len(participants)}")
    print(f"  {', '.join(participants)}")

    # Sequence length statistics
    seq_lengths = [trial["sequence_length"] for trial in dataset]
    print("\nSequence lengths:")
    print(f"  Min: {min(seq_lengths)}")
    print(f"  Max: {max(seq_lengths)}")
    print(f"  Mean: {sum(seq_lengths) / len(seq_lengths):.1f}")
    print(f"  Median: {sorted(seq_lengths)[len(seq_lengths) // 2]}")

    # Trials per participant
    print("\nTrials per participant:")
    for participant in participants:
        p = participant  # Bind to local variable to avoid closure issue
        count = len(dataset.filter(lambda x, p=p: x["participant"] == p))
        print(f"  {participant}: {count}")

    print("=" * 60 + "\n")


def clear_cache(cache_dir: str | Path) -> None:
    """
    Clear cached dataset files.

    Args:
        cache_dir: Cache directory to clear

    Example:
        >>> clear_cache("data/processed/phase1")
    """
    import shutil

    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        print(f"⚠️  Cache directory does not exist: {cache_dir}")
        return

    print(f"🗑️  Clearing cache at {cache_dir}...")

    try:
        # Remove all files and subdirectories
        for item in cache_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        print("✅ Cache cleared successfully")

    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")


def get_cache_info(cache_dir: str | Path) -> dict:
    """
    Get information about cached dataset.

    Args:
        cache_dir: Cache directory

    Returns:
        Dictionary with cache information

    Example:
        >>> info = get_cache_info("data/processed/phase1")
        >>> print(f"Cache size: {info['size_gb']:.2f} GB")
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return {
            "exists": False,
            "size_gb": 0,
            "num_files": 0,
            "metadata": None,
        }

    # Calculate total size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    num_files = len(list(cache_dir.rglob("*")))

    # Load metadata if exists
    metadata_file = cache_dir / "metadata.json"
    metadata = None
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
        except Exception:
            pass

    return {
        "exists": True,
        "size_gb": total_size / (1024**3),
        "num_files": num_files,
        "metadata": metadata,
        "path": str(cache_dir),
    }
