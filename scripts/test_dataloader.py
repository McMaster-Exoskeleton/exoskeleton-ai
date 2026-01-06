"""
Test script for ExoskeletonDataset and DataLoader.

This script demonstrates:
1. Loading dataset from HuggingFace (or local cache)
2. Creating train/val/test dataloaders
3. Iterating through batches
4. Inspecting data shapes and statistics

Usage:
    # After dataset is uploaded to HuggingFace
    python scripts/test_dataloader.py --hf-repo MacExo/exoData

    # Or test with local cache only (no download)
    python scripts/test_dataloader.py --no-download
"""

import argparse

from exoskeleton_ml.data import ExoskeletonDataset, create_dataloaders


def test_single_dataset(hf_repo: str, download: bool = True) -> None:
    """Test loading a single dataset."""
    print("=" * 60)
    print("TEST 1: Single Dataset")
    print("=" * 60)

    # Create dataset for a few participants
    dataset = ExoskeletonDataset(
        hf_repo=hf_repo,
        participants=["BT01", "BT02"],
        cache_dir="data/processed/phase1",
        download=download,
        normalize=False,
    )

    print(f"\n✅ Dataset loaded: {len(dataset)} trials")

    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Trials: {stats['num_trials']}")
    print(f"  Participants: {stats['participants']}")
    print(f"  Sequence lengths:")
    print(f"    Min: {stats['sequence_lengths']['min']}")
    print(f"    Max: {stats['sequence_lengths']['max']}")
    print(f"    Mean: {stats['sequence_lengths']['mean']:.1f}")

    # Test __getitem__
    print(f"\nSample trial (index 0):")
    sample = dataset[0]
    print(f"  Participant: {sample['participant']}")
    print(f"  Trial: {sample['trial_name']}")
    print(f"  Inputs shape: {sample['inputs'].shape}")
    print(f"  Targets shape: {sample['targets'].shape}")
    print(f"  Sequence length: {sample['sequence_length']}")


def test_dataloaders(hf_repo: str, download: bool = True) -> None:
    """Test creating dataloaders for train/val/test."""
    print("\n" + "=" * 60)
    print("TEST 2: DataLoaders")
    print("=" * 60)

    # Create dataloaders with participant splits
    train_loader, val_loader, test_loader = create_dataloaders(
        hf_repo=hf_repo,
        cache_dir="data/processed/phase1",
        train_participants=["BT01", "BT02", "BT03"],
        val_participants=["BT06"],
        test_participants=["BT07"],
        batch_size=4,
        num_workers=0,  # 0 for testing to avoid multiprocessing issues
        normalize=False,
    )

    print(f"\n✅ DataLoaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Test iteration
    print(f"\nTest iteration on train loader:")
    batch = next(iter(train_loader))
    print(f"  Batch inputs shape: {batch['inputs'].shape}")
    print(f"  Batch targets shape: {batch['targets'].shape}")
    print(f"  Batch lengths: {batch['lengths'].tolist()}")
    print(f"  Batch mask shape: {batch['mask'].shape}")
    print(f"  Participants: {batch['metadata']['participants']}")

    # Verify padding
    max_len = batch['inputs'].shape[1]
    print(f"\n  Max sequence length in batch: {max_len}")
    print(f"  Padding verification:")
    for i, length in enumerate(batch['lengths']):
        # Check if data after length is all zeros (padding)
        padding = batch['inputs'][i, length:].abs().sum().item()
        print(f"    Sample {i}: length={length}, padding_sum={padding:.2f}")


def test_normalization(hf_repo: str, download: bool = True) -> None:
    """Test dataset with normalization."""
    print("\n" + "=" * 60)
    print("TEST 3: Normalization")
    print("=" * 60)

    # Create normalized dataset
    dataset = ExoskeletonDataset(
        hf_repo=hf_repo,
        participants=["BT01"],
        cache_dir="data/processed/phase1",
        download=download,
        normalize=True,
    )

    print(f"\n✅ Normalized dataset created")

    # Check normalization stats exist
    if dataset.normalization_stats:
        print(f"\nNormalization stats computed:")
        print(f"  Inputs mean (first 5): {dataset.normalization_stats['inputs']['mean'][:5]}")
        print(f"  Inputs std (first 5): {dataset.normalization_stats['inputs']['std'][:5]}")
        print(f"  Targets mean: {dataset.normalization_stats['targets']['mean']}")
        print(f"  Targets std: {dataset.normalization_stats['targets']['std']}")

    # Get a sample and check it's normalized
    sample = dataset[0]
    print(f"\nSample after normalization:")
    print(f"  Inputs mean: {sample['inputs'].mean():.4f} (should be ~0)")
    print(f"  Inputs std: {sample['inputs'].std():.4f} (should be ~1)")


def test_cache_management() -> None:
    """Test cache management utilities."""
    print("\n" + "=" * 60)
    print("TEST 4: Cache Management")
    print("=" * 60)

    from exoskeleton_ml.data import get_cache_info

    cache_dir = "data/processed/phase1"
    info = get_cache_info(cache_dir)

    print(f"\nCache info for {cache_dir}:")
    print(f"  Exists: {info['exists']}")
    if info['exists']:
        print(f"  Size: {info['size_gb']:.2f} GB")
        print(f"  Files: {info['num_files']}")
        if info['metadata']:
            print(f"  Repository: {info['metadata']['hf_repo']}")
            print(f"  Participants: {info['metadata']['participants']}")
            print(f"  Total trials: {info['metadata']['total_trials']}")


def main() -> None:
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test ExoskeletonDataset and DataLoader")

    parser.add_argument(
        "--hf-repo",
        type=str,
        default="MacExo/exoData",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download from HuggingFace (use cached data only)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["dataset", "dataloader", "normalization", "cache", "all"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    download = not args.no_download

    try:
        if args.test in ["dataset", "all"]:
            test_single_dataset(args.hf_repo, download)

        if args.test in ["dataloader", "all"]:
            test_dataloaders(args.hf_repo, download)

        if args.test in ["normalization", "all"]:
            test_normalization(args.hf_repo, download)

        if args.test in ["cache", "all"]:
            test_cache_management()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
