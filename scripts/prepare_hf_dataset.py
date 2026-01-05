"""
Prepare exoskeleton data for HuggingFace upload.

Extracts required features from Phase1 data and converts to HF Dataset format.
Automatically converts to Parquet for efficient storage and loading.

Usage:
    python scripts/prepare_hf_dataset.py --input data/Phase1 --output data/hf_dataset
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


# Feature definitions based on 4-IMU, 4-motor setup
IMU_FEATURES = [
    # Left leg - Thigh IMU (6 features)
    "thigh_imu_l_accel_x",
    "thigh_imu_l_accel_y",
    "thigh_imu_l_accel_z",
    "thigh_imu_l_gyro_x",
    "thigh_imu_l_gyro_y",
    "thigh_imu_l_gyro_z",
    # Left leg - Shank IMU (6 features)
    "shank_imu_l_accel_x",
    "shank_imu_l_accel_y",
    "shank_imu_l_accel_z",
    "shank_imu_l_gyro_x",
    "shank_imu_l_gyro_y",
    "shank_imu_l_gyro_z",
    # Right leg - Thigh IMU (6 features)
    "thigh_imu_r_accel_x",
    "thigh_imu_r_accel_y",
    "thigh_imu_r_accel_z",
    "thigh_imu_r_gyro_x",
    "thigh_imu_r_gyro_y",
    "thigh_imu_r_gyro_z",
    # Right leg - Shank IMU (6 features)
    "shank_imu_r_accel_x",
    "shank_imu_r_accel_y",
    "shank_imu_r_accel_z",
    "shank_imu_r_gyro_x",
    "shank_imu_r_gyro_y",
    "shank_imu_r_gyro_z",
]

ANGLE_FEATURES = [
    "hip_flexion_l",  # Left hip angle (filtered)
    "hip_flexion_r",  # Right hip angle (filtered)
    "knee_angle_l",  # Left knee angle (filtered)
    "knee_angle_r",  # Right knee angle (filtered)
]

TARGET_FEATURES = [
    "hip_flexion_l_moment",  # Left hip biological moment (Nm/kg)
    "hip_flexion_r_moment",  # Right hip biological moment (Nm/kg)
    "knee_angle_l_moment",  # Left knee biological moment (Nm/kg)
    "knee_angle_r_moment",  # Right knee biological moment (Nm/kg)
]

# Participant masses (kg) from published paper
PARTICIPANT_MASSES = {
    "BT01": 80.59,
    "BT02": 72.24,
    "BT03": 95.29,
    "BT06": 79.33,
    "BT07": 64.49,
    "BT08": 69.13,
    "BT09": 82.31,
    "BT10": 93.45,
    "BT11": 50.39,
    "BT12": 78.15,
    "BT13": 89.85,
    "BT14": 67.30,
    "BT15": 58.40,
    "BT16": 64.33,
    "BT17": 60.03,
}


def process_trial(
    trial_dir: Path, participant: str, trial_name: str, validate: bool = False
) -> dict[str, Any] | None:
    """
    Process a single trial directory and extract required features.

    Args:
        trial_dir: Path to trial directory
        participant: Participant ID (e.g., "BT01")
        trial_name: Trial name (e.g., "normal_walk_1_1-2_on")
        validate: If True, perform extra validation checks

    Returns:
        Dictionary with trial data or None if files missing/invalid
    """
    # Construct expected file names
    base_name = f"{participant}_{trial_name}"
    exo_file = trial_dir / f"{base_name}_exo.csv"
    angle_file = trial_dir / f"{base_name}_angle_filt.csv"
    moment_file = trial_dir / f"{base_name}_moment_filt_bio.csv"

    # Check if all required files exist
    if not exo_file.exists():
        print(f"⚠️  Skipping {participant}/{trial_name} - missing exo.csv")
        return None
    if not angle_file.exists():
        print(f"⚠️  Skipping {participant}/{trial_name} - missing angle_filt.csv")
        return None
    if not moment_file.exists():
        print(f"⚠️  Skipping {participant}/{trial_name} - missing moment_filt_bio.csv")
        return None

    try:
        # Load CSVs
        exo_df = pd.read_csv(exo_file)
        angle_df = pd.read_csv(angle_file)
        moment_df = pd.read_csv(moment_file)

        # Validate shapes match
        if not (len(exo_df) == len(angle_df) == len(moment_df)):
            print(
                f"⚠️  Skipping {participant}/{trial_name} - length mismatch: "
                f"exo={len(exo_df)}, angle={len(angle_df)}, moment={len(moment_df)}"
            )
            return None

        # Check for required columns
        missing_imu = [col for col in IMU_FEATURES if col not in exo_df.columns]
        if missing_imu:
            print(
                f"⚠️  Skipping {participant}/{trial_name} - "
                f"missing IMU columns: {missing_imu}"
            )
            return None

        missing_angle = [col for col in ANGLE_FEATURES if col not in angle_df.columns]
        if missing_angle:
            print(
                f"⚠️  Skipping {participant}/{trial_name} - "
                f"missing angle columns: {missing_angle}"
            )
            return None

        missing_moment = [col for col in TARGET_FEATURES if col not in moment_df.columns]
        if missing_moment:
            print(
                f"⚠️  Skipping {participant}/{trial_name} - "
                f"missing moment columns: {missing_moment}"
            )
            return None

        # Extract features
        imu_data = exo_df[IMU_FEATURES].values  # Shape: (timesteps, 24)
        angle_data = angle_df[ANGLE_FEATURES].values  # Shape: (timesteps, 4)
        moment_data = moment_df[TARGET_FEATURES].values  # Shape: (timesteps, 4)

        # Validation checks
        if validate:
            # Check for NaN/inf values
            if pd.isna(imu_data).any():
                print(f"⚠️  Warning: {participant}/{trial_name} has NaN in IMU data")
            if pd.isna(angle_data).any():
                print(f"⚠️  Warning: {participant}/{trial_name} has NaN in angle data")
            if pd.isna(moment_data).any():
                print(f"⚠️  Warning: {participant}/{trial_name} has NaN in moment data")

            # Check for sequence length
            if len(exo_df) == 0:
                print(f"⚠️  Skipping {participant}/{trial_name} - empty sequence")
                return None

        # Combine into single record
        return {
            "participant": participant,
            "trial_name": trial_name,
            "mass_kg": float(PARTICIPANT_MASSES.get(participant, 0.0)),
            "sequence_length": int(len(exo_df)),
            # Store as lists (HF will handle serialization to Parquet)
            "imu_features": imu_data.tolist(),
            "angle_features": angle_data.tolist(),
            "moment_targets": moment_data.tolist(),
        }

    except Exception as e:
        print(f"❌ Error processing {participant}/{trial_name}: {e}")
        return None


def create_dataset_from_phase1(
    data_dir: str | Path, output_dir: str | Path | None = None, validate: bool = False
) -> Dataset:
    """
    Create HuggingFace Dataset from Phase1 data directory.

    Args:
        data_dir: Path to Phase1 directory containing participant folders
        output_dir: Optional path to save dataset locally
        validate: If True, perform extra validation checks

    Returns:
        HuggingFace Dataset object
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all participant directories
    participants = sorted(
        [p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("BT")]
    )

    if not participants:
        raise ValueError(f"No participant directories found in {data_dir}")

    print(f"Found {len(participants)} participants: {participants}")
    print("=" * 60)

    all_trials = []
    stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0}

    for participant in participants:
        participant_dir = data_dir / participant

        # Find all trial directories
        trials = sorted(
            [t.name for t in participant_dir.iterdir() if t.is_dir() and "." not in t.name]
        )

        stats["total"] += len(trials)
        print(f"\nProcessing {participant}: {len(trials)} trials")

        for trial_name in tqdm(trials, desc=f"{participant}", ncols=80):
            trial_dir = participant_dir / trial_name
            trial_data = process_trial(trial_dir, participant, trial_name, validate)

            if trial_data is not None:
                all_trials.append(trial_data)
                stats["success"] += 1
            else:
                stats["skipped"] += 1

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total trials found:        {stats['total']}")
    print(f"Successfully processed:    {stats['success']} ✅")
    print(f"Skipped (missing files):   {stats['skipped']} ⚠️")
    print(f"Errors:                    {stats['errors']} ❌")
    print(f"Success rate:              {stats['success']/stats['total']*100:.1f}%")

    if stats["success"] == 0:
        raise ValueError("No trials were successfully processed!")

    print(f"\n✅ Creating HuggingFace Dataset from {stats['success']} trials...")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(all_trials)

    # Optionally save locally (automatically uses Parquet)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving dataset to {output_dir}...")
        dataset.save_to_disk(str(output_dir))
        print(f"✅ Dataset saved successfully!")

        # Estimate disk usage
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        print(f"📊 Disk usage: {total_size / (1024**3):.2f} GB")

    return dataset


def print_dataset_info(dataset: Dataset) -> None:
    """Print dataset statistics and schema."""
    print("\n" + "=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    print(f"Total trials: {len(dataset)}")

    print(f"\n📋 Schema:")
    for key, feature in dataset.features.items():
        print(f"  • {key}: {feature}")

    # Participant distribution
    participants = dataset.unique("participant")
    print(f"\n👥 Participants ({len(participants)}):")
    for p in sorted(participants):
        count = len(dataset.filter(lambda x: x["participant"] == p))
        mass = PARTICIPANT_MASSES.get(p, 0.0)
        print(f"  • {p}: {count:3d} trials, {mass:.2f} kg")

    # Sample trial info
    print(f"\n📝 Example trial:")
    example = dataset[0]
    print(f"  • Participant: {example['participant']}")
    print(f"  • Trial: {example['trial_name']}")
    print(f"  • Mass: {example['mass_kg']} kg")
    print(f"  • Sequence length: {example['sequence_length']} timesteps")
    print(f"  • IMU features: ({example['sequence_length']}, 24)")
    print(f"  • Angle features: ({example['sequence_length']}, 4)")
    print(f"  • Moment targets: ({example['sequence_length']}, 4)")

    # Sequence length statistics
    seq_lengths = [trial["sequence_length"] for trial in dataset]
    print(f"\n📊 Sequence length statistics:")
    print(f"  • Min: {min(seq_lengths)}")
    print(f"  • Max: {max(seq_lengths)}")
    print(f"  • Mean: {sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"  • Median: {sorted(seq_lengths)[len(seq_lengths)//2]}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare exoskeleton Phase1 data for HuggingFace upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Phase1 data and save to local directory
  python scripts/prepare_hf_dataset.py --input data/Phase1 --output data/hf_dataset

  # Process with extra validation checks
  python scripts/prepare_hf_dataset.py --input data/Phase1 --output data/hf_dataset --validate

  # Process without saving (just preview)
  python scripts/prepare_hf_dataset.py --input data/Phase1 --no-save
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/Phase1",
        help="Path to Phase1 data directory (default: data/Phase1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/hf_dataset",
        help="Path to save processed dataset (default: data/hf_dataset)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform extra validation checks (slower but more thorough)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save dataset to disk (just preview processing)",
    )

    args = parser.parse_args()

    # Process dataset
    print("🚀 Starting data preparation for HuggingFace...")
    print(f"📂 Input directory: {args.input}")
    if not args.no_save:
        print(f"💾 Output directory: {args.output}")
    print(f"✓ Validation: {'enabled' if args.validate else 'disabled'}")
    print()

    try:
        dataset = create_dataset_from_phase1(
            data_dir=args.input,
            output_dir=None if args.no_save else args.output,
            validate=args.validate,
        )

        print_dataset_info(dataset)

        print("\n" + "=" * 60)
        print("✅ DATASET READY FOR UPLOAD TO HUGGINGFACE!")
        print("=" * 60)
        print("\n📤 Next steps:")
        print("  1. Review the dataset info above")
        print("  2. Test loading: ")
        print(f"     from datasets import load_from_disk")
        print(f"     dataset = load_from_disk('{args.output}')")
        print("  3. Upload to HuggingFace:")
        print("     python scripts/upload_to_hf.py")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
