"""Dataset classes for exoskeleton data."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .download import download_and_cache_dataset


class ExoskeletonDataset(Dataset):
    """PyTorch Dataset for exoskeleton movement data from HuggingFace.

    This dataset handles:
    - Auto-download from HuggingFace on first use
    - Local caching of preprocessed PyTorch tensors
    - Participant-based filtering for train/val/test splits
    - Variable-length sequence support
    - Optional normalization

    Example:
        >>> # Initialize with auto-download
        >>> dataset = ExoskeletonDataset(
        ...     hf_repo="MacExo/exoData",
        ...     participants=["BT01", "BT02", "BT03"],
        ...     cache_dir="data/processed/phase1"
        ... )
        >>>
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=8,
        ...     collate_fn=dataset.collate_fn,
        ...     shuffle=True
        ... )
        >>>
        >>> # Iterate
        >>> for batch in loader:
        ...     inputs = batch["inputs"]  # (batch, seq_len, 28)
        ...     targets = batch["targets"]  # (batch, seq_len, 4)
        ...     lengths = batch["lengths"]  # (batch,)
    """

    def __init__(
        self,
        hf_repo: str = "MacExo/exoData",
        cache_dir: str | Path = "data/processed/phase1",
        participants: list[str] | None = None,
        download: bool = True,
        force_redownload: bool = False,
        force_reprocess: bool = False,
        normalize: bool = False,
        normalization_stats: dict | None = None,
    ) -> None:
        """
        Initialize ExoskeletonDataset.

        Args:
            hf_repo: HuggingFace repository ID
            cache_dir: Local directory to cache preprocessed data
            participants: Optional list of participants to include (e.g., ["BT01", "BT02"])
                         If None, includes all participants
            download: Whether to auto-download from HF if not cached
            force_redownload: Force re-download from HF even if cached
            force_reprocess: Force reprocessing even if preprocessed cache exists
            normalize: Whether to normalize features
            normalization_stats: Optional dict with normalization statistics
                                (mean, std, min, max). If None and normalize=True,
                                will compute from data.
        """
        self.hf_repo = hf_repo
        self.cache_dir = Path(cache_dir)
        self.participants = participants
        self.normalize = normalize
        self.normalization_stats = normalization_stats

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset from HuggingFace (if needed)
        if download or force_redownload:
            self.hf_dataset = download_and_cache_dataset(
                hf_repo=hf_repo,
                cache_dir=cache_dir,
                force_redownload=force_redownload,
            )
        else:
            # Load from cache
            hf_cache_dir = self.cache_dir / "hf_cache"
            if not hf_cache_dir.exists():
                raise FileNotFoundError(
                    f"Dataset not found in cache: {hf_cache_dir}\n"
                    f"Set download=True to auto-download from HuggingFace"
                )
            from datasets import load_from_disk

            self.hf_dataset = load_from_disk(str(hf_cache_dir))

        # Filter by participants if specified
        if participants is not None:
            print(f"🔍 Filtering dataset for participants: {participants}")
            self.hf_dataset = self.hf_dataset.filter(lambda x: x["participant"] in participants)
            print(f"   Filtered to {len(self.hf_dataset)} trials")

        # Check cache and process if needed
        self._setup_cache(force_reprocess)

        # Compute normalization stats if needed
        if self.normalize and self.normalization_stats is None:
            print("📊 Computing normalization statistics...")
            self.normalization_stats = self._compute_normalization_stats()
            # Save stats
            stats_file = self.cache_dir / "normalization_stats.json"
            with open(stats_file, "w") as f:
                json.dump(self.normalization_stats, f, indent=2)
            print(f"   Saved normalization stats to {stats_file}")

    def _setup_cache(self, force_reprocess: bool = False) -> None:
        """Setup preprocessed cache or load existing cache."""
        # Cache key based on participants (to support different splits)
        participant_key = "_".join(sorted(self.participants)) if self.participants else "all"
        cache_key = hashlib.md5(participant_key.encode()).hexdigest()[:8]

        self.preprocessed_dir = self.cache_dir / f"preprocessed_{cache_key}"
        self.index_file = self.preprocessed_dir / "index.pkl"

        # Check if cache exists
        if self.index_file.exists() and not force_reprocess:
            print("✅ Loading preprocessed data from cache...")
            with open(self.index_file, "rb") as f:
                self.index = pickle.load(f)
            print(f"   Loaded {len(self.index)} preprocessed trials")
        else:
            print("🔄 Preprocessing trials and caching...")
            self._preprocess_and_cache()

    def _preprocess_and_cache(self) -> None:
        """Preprocess all trials and cache as PyTorch tensors."""
        from tqdm import tqdm

        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        self.index = []

        for idx, trial in enumerate(tqdm(self.hf_dataset, desc="Preprocessing")):
            # Convert to tensors
            imu_features = torch.tensor(trial["imu_features"], dtype=torch.float32)
            angle_features = torch.tensor(trial["angle_features"], dtype=torch.float32)
            moment_targets = torch.tensor(trial["moment_targets"], dtype=torch.float32)

            # Concatenate inputs: (seq_len, 28)
            inputs = torch.cat([imu_features, angle_features], dim=-1)
            targets = moment_targets

            # Save to disk
            trial_file = self.preprocessed_dir / f"trial_{idx:05d}.pt"
            torch.save(
                {
                    "inputs": inputs,
                    "targets": targets,
                    "participant": trial["participant"],
                    "trial_name": trial["trial_name"],
                    "mass_kg": trial["mass_kg"],
                    "sequence_length": trial["sequence_length"],
                },
                trial_file,
            )

            # Add to index
            self.index.append(
                {
                    "file": str(trial_file),
                    "participant": trial["participant"],
                    "trial_name": trial["trial_name"],
                    "sequence_length": trial["sequence_length"],
                }
            )

        # Save index
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)

        print(f"✅ Preprocessed and cached {len(self.index)} trials")

        # Calculate and report cache size
        total_size = sum(
            Path(item["file"]).stat().st_size for item in self.index if Path(item["file"]).exists()
        )
        print(f"   Cache size: {total_size / (1024**3):.2f} GB")

    def _compute_normalization_stats(self) -> dict:
        """Compute normalization statistics (mean, std) across all trials.

        Note: Handles NaN values by using nanmean/nanstd which ignore NaN values
        when computing statistics.
        """
        # Accumulate statistics
        inputs_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []

        for idx in range(len(self)):
            trial_data = torch.load(self.index[idx]["file"])
            inputs_list.append(trial_data["inputs"])
            targets_list.append(trial_data["targets"])

        # Concatenate all data
        all_inputs: torch.Tensor = torch.cat(inputs_list, dim=0)  # (total_timesteps, 28)
        all_targets: torch.Tensor = torch.cat(targets_list, dim=0)  # (total_timesteps, 4)

        # Compute statistics (using nanmean/nanstd to handle NaN values)
        stats = {
            "inputs": {
                "mean": torch.nanmean(all_inputs, dim=0).tolist(),
                "std": [
                    (
                        all_inputs[:, i][~torch.isnan(all_inputs[:, i])].std().item()
                        if (~torch.isnan(all_inputs[:, i])).sum() > 1
                        else 1.0
                    )  # Default to 1.0 if all values are NaN
                    for i in range(all_inputs.shape[1])
                ],
                "min": [
                    (
                        all_inputs[:, i][~torch.isnan(all_inputs[:, i])].min().item()
                        if (~torch.isnan(all_inputs[:, i])).any()
                        else 0.0
                    )
                    for i in range(all_inputs.shape[1])
                ],
                "max": [
                    (
                        all_inputs[:, i][~torch.isnan(all_inputs[:, i])].max().item()
                        if (~torch.isnan(all_inputs[:, i])).any()
                        else 1.0
                    )
                    for i in range(all_inputs.shape[1])
                ],
            },
            "targets": {
                "mean": torch.nanmean(all_targets, dim=0).tolist(),
                "std": [
                    (
                        all_targets[:, i][~torch.isnan(all_targets[:, i])].std().item()
                        if (~torch.isnan(all_targets[:, i])).sum() > 1
                        else 1.0
                    )
                    for i in range(all_targets.shape[1])
                ],
                "min": [
                    (
                        all_targets[:, i][~torch.isnan(all_targets[:, i])].min().item()
                        if (~torch.isnan(all_targets[:, i])).any()
                        else 0.0
                    )
                    for i in range(all_targets.shape[1])
                ],
                "max": [
                    (
                        all_targets[:, i][~torch.isnan(all_targets[:, i])].max().item()
                        if (~torch.isnan(all_targets[:, i])).any()
                        else 1.0
                    )
                    for i in range(all_targets.shape[1])
                ],
            },
        }

        return stats

    def __len__(self) -> int:
        """Return the number of trials in the dataset."""
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single trial.

        Args:
            idx: Trial index

        Returns:
            Dictionary containing:
                - inputs: (seq_len, 28) tensor
                - targets: (seq_len, 4) tensor
                - participant: str
                - trial_name: str
                - mass_kg: float
                - sequence_length: int
        """
        # Load preprocessed trial
        trial_data: dict[str, Any] = torch.load(self.index[idx]["file"])

        # Apply normalization if enabled
        if self.normalize and self.normalization_stats is not None:
            inputs_mean = torch.tensor(self.normalization_stats["inputs"]["mean"])
            inputs_std = torch.tensor(self.normalization_stats["inputs"]["std"])
            trial_data["inputs"] = (trial_data["inputs"] - inputs_mean) / (inputs_std + 1e-8)

            targets_mean = torch.tensor(self.normalization_stats["targets"]["mean"])
            targets_std = torch.tensor(self.normalization_stats["targets"]["std"])
            trial_data["targets"] = (trial_data["targets"] - targets_mean) / (targets_std + 1e-8)

        return trial_data

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """
        Collate function for variable-length sequences.

        Pads sequences to the maximum length in the batch.

        Args:
            batch: List of trial dictionaries from __getitem__

        Returns:
            Dictionary containing:
                - inputs: (batch_size, max_seq_len, 28) padded tensor
                - targets: (batch_size, max_seq_len, 4) padded tensor
                - lengths: (batch_size,) tensor with actual sequence lengths
                - mask: (batch_size, max_seq_len) tensor (1=real, 0=padding)
                - metadata: dict with participant info, trial names, masses
        """
        batch_size = len(batch)
        max_len = max(item["sequence_length"] for item in batch)

        # Pre-allocate tensors
        inputs = torch.zeros(batch_size, max_len, 28)
        targets = torch.zeros(batch_size, max_len, 4)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        # Metadata
        participants = []
        trial_names = []
        masses = []

        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = item["sequence_length"]
            inputs[i, :seq_len] = item["inputs"]
            targets[i, :seq_len] = item["targets"]
            lengths[i] = seq_len
            mask[i, :seq_len] = True

            participants.append(item["participant"])
            trial_names.append(item["trial_name"])
            masses.append(item["mass_kg"])

        return {
            "inputs": inputs,
            "targets": targets,
            "lengths": lengths,
            "mask": mask,
            "metadata": {
                "participants": participants,
                "trial_names": trial_names,
                "masses": masses,
            },
        }

    def get_trial_info(self, idx: int) -> dict[str, Any]:
        """
        Get metadata for a specific trial without loading full data.

        Args:
            idx: Trial index

        Returns:
            Dictionary with trial metadata
        """
        return dict(self.index[idx])

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        seq_lengths = [item["sequence_length"] for item in self.index]
        participants = list({item["participant"] for item in self.index})

        return {
            "num_trials": len(self),
            "num_participants": len(participants),
            "participants": sorted(participants),
            "sequence_lengths": {
                "min": min(seq_lengths),
                "max": max(seq_lengths),
                "mean": sum(seq_lengths) / len(seq_lengths),
                "median": sorted(seq_lengths)[len(seq_lengths) // 2],
            },
        }


def create_dataloaders(
    hf_repo: str = "MacExo/exoData",
    cache_dir: str | Path = "data/processed/phase1",
    train_participants: list[str] | None = None,
    val_participants: list[str] | None = None,
    test_participants: list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        hf_repo: HuggingFace repository ID
        cache_dir: Cache directory
        train_participants: List of participant IDs for training
        val_participants: List of participant IDs for validation
        test_participants: List of participant IDs for testing
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        normalize: Whether to normalize features
        **kwargs: Additional arguments for DataLoader

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_participants=["BT01", "BT02", "BT03"],
        ...     val_participants=["BT14"],
        ...     test_participants=["BT15", "BT16"],
        ...     batch_size=16
        ... )
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = ExoskeletonDataset(
        hf_repo=hf_repo,
        cache_dir=cache_dir,
        participants=train_participants,
        normalize=normalize,
    )

    # Use train dataset's normalization stats for val and test
    normalization_stats = train_dataset.normalization_stats if normalize else None

    val_dataset = (
        ExoskeletonDataset(
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            participants=val_participants,
            normalize=normalize,
            normalization_stats=normalization_stats,
        )
        if val_participants
        else None
    )

    test_dataset = (
        ExoskeletonDataset(
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            participants=test_participants,
            normalize=normalize,
            normalization_stats=normalization_stats,
        )
        if test_participants
        else None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        **kwargs,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
            **kwargs,
        )
        if val_dataset
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            **kwargs,
        )
        if test_dataset
        else None
    )

    return train_loader, val_loader, test_loader
