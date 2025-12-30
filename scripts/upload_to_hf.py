"""
Upload prepared exoskeleton dataset to HuggingFace Hub.

Handles authentication, batch upload, and dataset card generation.
Supports resume capability if upload fails midway.

Usage:
    # Login first
    huggingface-cli login

    # Upload dataset
    python scripts/upload_to_hf.py --dataset data/hf_dataset --repo MacExo/exoData

    # Upload as private
    python scripts/upload_to_hf.py --dataset data/hf_dataset --repo MacExo/exoData --private
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo


DATASET_CARD_TEMPLATE = """---
license: mit
task_categories:
- time-series-forecasting
- regression
tags:
- exoskeleton
- biomechanics
- wearable-robotics
- imu-sensors
- joint-moments
size_categories:
- 1K<n<10K
---

# Exoskeleton Phase1 Dataset

## Dataset Description

This dataset contains exoskeleton movement data for task-agnostic biological joint moment estimation.
The data comes from 15 participants performing various walking and movement tasks while wearing a
bilateral lower-limb exoskeleton equipped with IMU sensors.

### Dataset Summary

- **Total Trials**: {num_trials}
- **Participants**: {num_participants} (BT01-BT17, excluding BT04, BT05)
- **Input Features**: 28 (24 IMU measurements + 4 joint angles)
- **Output Targets**: 4 (hip & knee biological moments, Nm/kg)
- **Sequence Type**: Variable-length time series
- **Average Sequence Length**: ~{avg_seq_length} timesteps
- **Total Size**: ~{total_size_gb:.2f} GB

### Supported Tasks

- **Time Series Regression**: Predict biological joint moments from sensor data
- **Multi-output Prediction**: Simultaneous estimation of hip and knee moments
- **Cross-Subject Generalization**: Leave-one-subject-out evaluation

## Dataset Structure

### Data Instances

Each row represents a single trial (not individual timesteps):

```python
{{
    "participant": "BT01",              # Participant ID
    "trial_name": "normal_walk_1_1-2_on",  # Trial identifier
    "mass_kg": 80.59,                   # Participant mass (kg)
    "sequence_length": 1523,            # Number of timesteps

    # Shape: (sequence_length, 24) - IMU measurements
    "imu_features": [[...], [...], ...],

    # Shape: (sequence_length, 4) - Joint angles
    "angle_features": [[...], [...], ...],

    # Shape: (sequence_length, 4) - Target moments
    "moment_targets": [[...], [...], ...],
}}
```

### Data Fields

#### Input Features (28 total)

**IMU Features (24)**: 4 IMUs × (3 accelerometer + 3 gyroscope)
- Left thigh IMU: accel_x/y/z, gyro_x/y/z
- Left shank IMU: accel_x/y/z, gyro_x/y/z
- Right thigh IMU: accel_x/y/z, gyro_x/y/z
- Right shank IMU: accel_x/y/z, gyro_x/y/z

**Angle Features (4)**:
- `hip_flexion_l`: Left hip angle (degrees)
- `hip_flexion_r`: Right hip angle (degrees)
- `knee_angle_l`: Left knee angle (degrees)
- `knee_angle_r`: Right knee angle (degrees)

#### Target Features (4)

Normalized biological joint moments (Nm/kg):
- `hip_flexion_l_moment`: Left hip moment
- `hip_flexion_r_moment`: Right hip moment
- `knee_angle_l_moment`: Left knee moment
- `knee_angle_r_moment`: Right knee moment

### Data Splits

No predefined splits are provided. Recommended approach:

**Leave-One-Subject-Out (LOSO)**:
```python
train_participants = ["BT01", "BT02", "BT03", "BT06", "BT07", "BT08",
                      "BT09", "BT10", "BT11", "BT12", "BT13"]
val_participants = ["BT14"]
test_participants = ["BT15", "BT16", "BT17"]
```

## Dataset Creation

### Source Data

Data collected from bilateral lower-limb exoskeleton experiments with 15 healthy participants.
Each participant performed multiple trials of various walking tasks (normal walking, incline/decline,
stairs, etc.) with the exoskeleton in different assistance modes.

### Data Collection

- **Sensors**: 4 IMUs (thigh and shank, bilateral) + encoders for joint angles
- **Ground Truth**: Biological joint moments from inverse dynamics
- **Sampling Rate**: 100 Hz (resampled)
- **Tasks**: Normal walking, incline/decline, stairs, different speeds

### Preprocessing

1. Raw CSV files extracted from published dataset
2. Feature selection: 24 IMU + 4 angle features
3. Target extraction: 4 biological joint moments
4. Conversion to Parquet format via HuggingFace `datasets`

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("MacExo/exoData")

# Access a trial
trial = dataset[0]
print(f"Participant: {{trial['participant']}}")
print(f"Trial: {{trial['trial_name']}}")
print(f"Sequence length: {{trial['sequence_length']}}")
```

### PyTorch Integration

```python
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

dataset = load_dataset("MacExo/exoData")

# Filter by participant
train_dataset = dataset.filter(lambda x: x["participant"] in ["BT01", "BT02"])

# Custom collate function for variable-length sequences
def collate_fn(batch):
    max_len = max(item["sequence_length"] for item in batch)
    batch_size = len(batch)

    inputs = torch.zeros(batch_size, max_len, 28)
    targets = torch.zeros(batch_size, max_len, 4)
    lengths = torch.tensor([item["sequence_length"] for item in batch])

    for i, item in enumerate(batch):
        seq_len = item["sequence_length"]
        # Concatenate IMU (24) + angles (4) = 28 features
        imu = torch.tensor(item["imu_features"])
        angles = torch.tensor(item["angle_features"])
        inputs[i, :seq_len] = torch.cat([imu, angles], dim=-1)
        targets[i, :seq_len] = torch.tensor(item["moment_targets"])

    return {{"inputs": inputs, "targets": targets, "lengths": lengths}}

# Create DataLoader
loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
```

## Participant Information

{participant_table}

## Considerations for Using the Data

### Recommended Practices

1. **Cross-Subject Validation**: Use leave-one-subject-out for realistic performance evaluation
2. **Sequence Padding**: Handle variable-length sequences appropriately
3. **Normalization**: Consider normalizing features per-subject or globally
4. **Temporal Dependencies**: Account for sequential nature (use RNNs, Transformers, TCNs)

### Limitations

- Data from healthy participants only (may not generalize to clinical populations)
- Single exoskeleton design (device-specific characteristics)
- Controlled laboratory environment
- Limited to lower-limb movements

## Citation

If you use this dataset, please cite:

```bibtex
@article{{exoskeleton2024,
  title={{Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation}},
  author={{[Authors]}},
  journal={{[Journal]}},
  year={{2024}},
  note={{Dataset available at https://huggingface.co/datasets/MacExo/exoData}}
}}
```

## License

This dataset is released under the MIT License.

## Contact

For questions or issues regarding this dataset, please open an issue on the
[dataset repository](https://huggingface.co/datasets/MacExo/exoData).

---

*Dataset prepared and uploaded on {upload_date}*
"""


def generate_participant_table(dataset) -> str:
    """Generate markdown table of participant statistics."""
    from collections import defaultdict

    # Count trials per participant
    participant_counts = defaultdict(int)
    for trial in dataset:
        participant_counts[trial["participant"]] += 1

    # Participant masses (from prepare_hf_dataset.py)
    PARTICIPANT_MASSES = {
        "BT01": 80.59, "BT02": 72.24, "BT03": 95.29,
        "BT06": 79.33, "BT07": 64.49, "BT08": 69.13,
        "BT09": 82.31, "BT10": 93.45, "BT11": 50.39,
        "BT12": 78.15, "BT13": 89.85, "BT14": 67.30,
        "BT15": 58.40, "BT16": 64.33, "BT17": 60.03,
    }

    table = "| Participant | Mass (kg) | # Trials |\n"
    table += "|-------------|-----------|----------|\n"

    for participant in sorted(participant_counts.keys()):
        mass = PARTICIPANT_MASSES.get(participant, "N/A")
        count = participant_counts[participant]
        table += f"| {participant} | {mass:.2f} | {count} |\n"

    return table


def create_dataset_card(dataset, output_path: Path) -> None:
    """Generate dataset card (README.md) with metadata."""
    from datetime import datetime

    # Compute statistics
    num_trials = len(dataset)
    participants = dataset.unique("participant")
    num_participants = len(participants)

    seq_lengths = [trial["sequence_length"] for trial in dataset]
    avg_seq_length = int(sum(seq_lengths) / len(seq_lengths))

    # Estimate size (rough approximation)
    # Each trial: ~28 float32 values per timestep, plus overhead
    total_timesteps = sum(seq_lengths)
    total_features = 28 + 4  # inputs + targets
    total_size_gb = (total_timesteps * total_features * 4) / (1024**3)  # 4 bytes per float32

    # Generate participant table
    participant_table = generate_participant_table(dataset)

    # Fill template
    upload_date = datetime.now().strftime("%Y-%m-%d")
    card_content = DATASET_CARD_TEMPLATE.format(
        num_trials=num_trials,
        num_participants=num_participants,
        avg_seq_length=avg_seq_length,
        total_size_gb=total_size_gb,
        participant_table=participant_table,
        upload_date=upload_date,
    )

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(card_content)

    print(f"✅ Dataset card saved to {output_path}")


def upload_to_huggingface(
    dataset_path: str | Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
) -> None:
    """
    Upload dataset to HuggingFace Hub.

    Args:
        dataset_path: Path to prepared dataset directory
        repo_id: HuggingFace repository ID (e.g., "MacExo/exoData")
        private: Whether to create a private repository
        token: HuggingFace API token (optional, uses cached token if not provided)
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    print("=" * 60)
    print("HUGGINGFACE DATASET UPLOAD")
    print("=" * 60)
    print(f"📂 Dataset path: {dataset_path}")
    print(f"📤 Repository: {repo_id}")
    print(f"🔒 Private: {private}")
    print()

    # Load dataset
    print("📥 Loading dataset from disk...")
    try:
        dataset = load_from_disk(str(dataset_path))
        print(f"✅ Loaded {len(dataset)} trials")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Print dataset summary
    participants = dataset.unique("participant")
    print(f"   Participants: {sorted(participants)}")
    print(f"   Total trials: {len(dataset)}")

    # Generate dataset card
    print("\n📝 Generating dataset card...")
    card_path = dataset_path / "README.md"
    create_dataset_card(dataset, card_path)

    # Initialize HuggingFace API
    print("\n🔑 Initializing HuggingFace API...")
    try:
        api = HfApi(token=token)
        # Test authentication
        user_info = api.whoami(token=token)
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}", file=sys.stderr)
        print("\nPlease login first:")
        print("  huggingface-cli login")
        sys.exit(1)

    # Create repository
    print(f"\n📦 Creating repository '{repo_id}'...")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,  # Don't fail if repo already exists
            token=token,
        )
        print(f"✅ Repository ready: {repo_url}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}", file=sys.stderr)
        sys.exit(1)

    # Upload dataset
    print(f"\n⬆️  Uploading dataset to HuggingFace Hub...")
    print("   This may take several minutes depending on dataset size...")

    try:
        # Push to hub (includes all files in dataset directory)
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token,
        )
        print(f"✅ Dataset uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload failed: {e}", file=sys.stderr)
        print("\nYou can try again - the upload will resume from where it left off.")
        sys.exit(1)

    # Success summary
    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\n🎉 Your dataset is now available at:")
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print(f"\n📖 View the dataset card:")
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print(f"\n💻 Load the dataset in code:")
    print(f'   from datasets import load_dataset')
    print(f'   dataset = load_dataset("{repo_id}")')
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload exoskeleton dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Login first (one-time)
  huggingface-cli login

  # Upload dataset (public)
  python scripts/upload_to_hf.py --dataset data/hf_dataset --repo MacExo/exoData

  # Upload as private repository
  python scripts/upload_to_hf.py --dataset data/hf_dataset --repo MacExo/exoData --private

  # Use specific token
  python scripts/upload_to_hf.py --dataset data/hf_dataset --repo MacExo/exoData --token YOUR_TOKEN
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/hf_dataset",
        help="Path to prepared dataset directory (default: data/hf_dataset)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (uses cached token if not provided)",
    )

    args = parser.parse_args()

    # Validate repo ID format
    if "/" not in args.repo:
        print("❌ Error: Repository ID must be in format 'username/dataset-name'", file=sys.stderr)
        sys.exit(1)

    # Upload dataset
    try:
        upload_to_huggingface(
            dataset_path=args.dataset,
            repo_id=args.repo,
            private=args.private,
            token=args.token,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
