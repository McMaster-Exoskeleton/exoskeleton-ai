# Data Infrastructure Implementation Plan

**Project**: Exoskeleton AI - Phase1 Data Pipeline to HuggingFace
**Goal**: Create a robust data infrastructure for storing, downloading, and loading exoskeleton training data
**Date**: 2025-12-23
**Status**: Planning

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Specifications](#data-specifications)
4. [Implementation Tasks](#implementation-tasks)
5. [Testing & Validation](#testing--validation)
6. [Usage Patterns](#usage-patterns)
7. [Future Considerations](#future-considerations)

---

## Overview

### Context
- We have 42GB of exoskeleton movement data from a published paper (Phase1 dataset)
- Data is currently in 25,035 CSV files across 15 participants (BT01-BT17)
- Each participant has ~100+ trials of different movement tasks
- Each trial has 10+ CSV files with different measurements

### Goals
1. **Upload to HuggingFace**: Store dataset in HF for team sharing and version control
2. **Efficient Format**: Convert to Parquet for faster loading and smaller storage
3. **Auto-Download**: Training code automatically downloads data on first use
4. **Local Caching**: After first download, data is cached locally
5. **PyTorch DataLoader**: Seamless integration with PyTorch training loops

### Non-Goals
- Splitting train/val/test on HuggingFace (will be done in training code)
- Processing additional phases beyond Phase1 (future work)
- Real-time data streaming (will batch load)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        HuggingFace Hub                          │
│                  (macexo/exoskeleton-phase1)                    │
│                    [Parquet Files + Metadata]                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ datasets.load_dataset()
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              ~/.cache/huggingface/datasets/                     │
│                    [Auto-cached by HF]                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ First-time: preprocess & cache
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              data/processed/phase1/                             │
│         [Preprocessed tensors for fast training]                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ torch.utils.data.DataLoader
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Training Loop                              │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Layer Caching Strategy

**Layer 1: HuggingFace Cache** (automatic)
- Location: `~/.cache/huggingface/datasets/`
- Contains: Raw Parquet files from HF
- Managed by: HuggingFace `datasets` library
- Purpose: Avoid re-downloading from internet

**Layer 2: Preprocessed Cache** (we implement)
- Location: `data/processed/phase1/`
- Contains: PyTorch-ready tensors (.pt files or HDF5)
- Managed by: Our code
- Purpose: Avoid re-preprocessing CSVs every training run

---

## Data Specifications

### Required Files Per Trial

For each trial (e.g., `BT01/normal_walk_1_1-2_on/`), we need exactly **3 CSV files**:

#### 1. Input Features: `{participant}_{trial}_exo.csv`
**Columns to extract (24 features):**
```python
IMU_FEATURES = [
    # Left leg - Thigh IMU (6)
    "thigh_imu_l_accel_x", "thigh_imu_l_accel_y", "thigh_imu_l_accel_z",
    "thigh_imu_l_gyro_x", "thigh_imu_l_gyro_y", "thigh_imu_l_gyro_z",

    # Left leg - Shank IMU (6)
    "shank_imu_l_accel_x", "shank_imu_l_accel_y", "shank_imu_l_accel_z",
    "shank_imu_l_gyro_x", "shank_imu_l_gyro_y", "shank_imu_l_gyro_z",

    # Right leg - Thigh IMU (6)
    "thigh_imu_r_accel_x", "thigh_imu_r_accel_y", "thigh_imu_r_accel_z",
    "thigh_imu_r_gyro_x", "thigh_imu_r_gyro_y", "thigh_imu_r_gyro_z",

    # Right leg - Shank IMU (6)
    "shank_imu_r_accel_x", "shank_imu_r_accel_y", "shank_imu_r_accel_z",
    "shank_imu_r_gyro_x", "shank_imu_r_gyro_y", "shank_imu_r_gyro_z",
]
```

#### 2. Additional Kinematics: `{participant}_{trial}_angle_filt.csv`
**Columns to extract (4 features):**
```python
ANGLE_FEATURES = [
    "hip_flexion_l",   # Left hip angle (filtered)
    "hip_flexion_r",   # Right hip angle (filtered)
    "knee_angle_l",    # Left knee angle (filtered)
    "knee_angle_r",    # Right knee angle (filtered)
]
```

#### 3. Target Labels: `{participant}_{trial}_moment_filt_bio.csv`
**Columns to extract (4 targets):**
```python
TARGET_FEATURES = [
    "hip_flexion_l",   # Left hip biological moment (Nm/kg)
    "hip_flexion_r",   # Right hip biological moment (Nm/kg)
    "knee_angle_l",    # Left knee biological moment (Nm/kg)
    "knee_angle_r",    # Right knee biological moment (Nm/kg)
]
```

### Total Feature Count
- **Input Features**: 28 (24 IMU + 4 angles)
- **Output Targets**: 4 (joint moments)
- **Metadata**: participant ID, trial name, mass, sequence length

### Data Format in HuggingFace

Each row in the HF dataset represents **one trial** (not one timestep):

```python
{
    "participant": "BT01",              # str
    "trial_name": "normal_walk_1_1-2_on",  # str
    "mass_kg": 80.59,                   # float
    "sequence_length": 1523,            # int (timesteps in trial)

    # Shape: (sequence_length, 24)
    "imu_features": [[...], [...], ...],     # List[List[float]]

    # Shape: (sequence_length, 4)
    "angle_features": [[...], [...], ...],   # List[List[float]]

    # Shape: (sequence_length, 4)
    "moment_targets": [[...], [...], ...],   # List[List[float]]
}
```

**Rationale**: Storing trials as rows (not individual timesteps) because:
- Preserves temporal structure
- Allows variable-length sequences
- Easier to implement participant-level train/val/test splits
- More efficient storage in Parquet

### Participant Metadata

```python
PARTICIPANT_MASSES = {
    "BT01": 80.59,  # kg
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
```

**Note**: Masses are required for potential normalization/denormalization of forces/moments.

---

## Implementation Tasks

### Task 1: Data Preparation Script

**File**: `scripts/prepare_hf_dataset.py`

**Responsibilities**:
1. Scan `data/Phase1/` directory for all participants and trials
2. For each trial, load the 3 required CSV files
3. Extract specified columns from each CSV
4. Validate data integrity (same number of timesteps, no NaNs)
5. Combine into HuggingFace Dataset format
6. Save locally as Parquet (via HF's `save_to_disk()`)

**Inputs**:
- `data/Phase1/` directory containing participant folders

**Outputs**:
- `data/hf_dataset/` directory with Parquet files
- Console logs showing progress and any errors

**Error Handling**:
- Skip trials with missing CSV files (log warning)
- Skip trials with mismatched sequence lengths (log warning)
- Skip trials with missing required columns (log error)
- Continue processing other trials even if one fails

**Validation Checks**:
- All 3 CSV files exist for each trial
- All required columns present in each CSV
- Sequence lengths match across all 3 CSVs
- No NaN/inf values in extracted data
- Participant ID is in known list

**Performance Considerations**:
- Process participants in parallel if possible
- Show progress bar with `tqdm`
- Estimate final dataset size
- Report processing statistics (success/skip/error counts)

**Example CLI Usage**:
```bash
python scripts/prepare_hf_dataset.py \
    --input data/Phase1 \
    --output data/hf_dataset \
    --validate  # optional: extra validation checks
```

---

### Task 2: HuggingFace Upload Script

**File**: `scripts/upload_to_hf.py`

**Responsibilities**:
1. Load dataset from `data/hf_dataset/`
2. Upload to HuggingFace Hub in batches (participant-by-participant)
3. Handle authentication
4. Provide resume capability if upload fails midway
5. Generate dataset card (README) with metadata

**Inputs**:
- `data/hf_dataset/` (prepared dataset)
- HF repository name (e.g., `macexo/exoskeleton-phase1`)
- HF authentication token (via `huggingface-cli login` or env var)

**Outputs**:
- Dataset published to HuggingFace Hub
- Dataset card (README.md) with description, features, usage

**Batch Upload Strategy**:
```python
# Option 1: Upload all at once (simple)
dataset.push_to_hub("macexo/exoskeleton-phase1")

# Option 2: Upload by participant (resumable)
for participant in participants:
    subset = dataset.filter(lambda x: x["participant"] == participant)
    # Upload to separate branch, then merge
    subset.push_to_hub(
        "macexo/exoskeleton-phase1",
        split=participant,  # Creates a separate split
    )
```

**Authentication**:
- Use `huggingface_hub.HfApi()` with token
- Token from: `~/.huggingface/token` or `HF_TOKEN` env var
- Require write access to repository

**Dataset Card Template**:
```markdown
# Exoskeleton Phase1 Dataset

## Dataset Description
Exoskeleton movement data for task-agnostic biological joint moment estimation.

## Dataset Structure
- **Total Trials**: ~1900 trials
- **Participants**: 15 (BT01-BT17)
- **Input Features**: 28 (24 IMU + 4 joint angles)
- **Output Targets**: 4 (hip & knee moments, Nm/kg)

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("macexo/exoskeleton-phase1")
```

## Citation
[Paper citation here]
```

**Example CLI Usage**:
```bash
# First login
huggingface-cli login

# Upload dataset
python scripts/upload_to_hf.py \
    --dataset data/hf_dataset \
    --repo macexo/exoskeleton-phase1 \
    --private  # optional: make repo private
```

---

### Task 3: PyTorch Dataset Class

**File**: `src/exoskeleton_ml/data/datasets.py`

**Class**: `ExoskeletonDataset(torch.utils.data.Dataset)`

**Responsibilities**:
1. Download data from HuggingFace on first use (auto-cached)
2. Preprocess trials into PyTorch tensors
3. Cache preprocessed tensors locally
4. Support flexible indexing (by trial, by participant, by task type)
5. Support train/val/test splitting by participant

**Interface**:
```python
from exoskeleton_ml.data import ExoskeletonDataset

# Initialize (downloads from HF if not cached)
dataset = ExoskeletonDataset(
    hf_repo="macexo/exoskeleton-phase1",
    cache_dir="data/processed/phase1",
    split="train",  # or "val", "test", None
    participants=["BT01", "BT02", ...],  # optional filter
    download=True,  # auto-download if not cached
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for batch in loader:
    inputs = batch["inputs"]     # (batch, seq_len, 28)
    targets = batch["targets"]   # (batch, seq_len, 4)
    lengths = batch["lengths"]   # (batch,) - actual seq lengths
    metadata = batch["metadata"] # dict with participant, trial_name
```

**Implementation Details**:

**Initialization**:
```python
def __init__(
    self,
    hf_repo: str = "macexo/exoskeleton-phase1",
    cache_dir: str | Path = "data/processed/phase1",
    split: str | None = None,
    participants: List[str] | None = None,
    download: bool = True,
    force_reprocess: bool = False,
):
    """
    Args:
        hf_repo: HuggingFace repository ID
        cache_dir: Local directory to cache preprocessed data
        split: Optional split name for predefined splits
        participants: Optional list of participants to include
        download: Whether to auto-download from HF if not cached
        force_reprocess: Force reprocessing even if cache exists
    """
```

**Caching Logic**:
```python
cache_dir/
├── metadata.json          # Dataset info, participants, trial counts
├── BT01/
│   ├── trial_001.pt       # Single trial as tensor
│   ├── trial_002.pt
│   └── ...
├── BT02/
│   └── ...
└── index.pkl              # Fast lookup: trial_id -> file_path
```

**Preprocessing**:
- Convert lists → numpy arrays → torch tensors
- Dtype: `torch.float32`
- Keep on CPU until loaded into DataLoader
- Optionally normalize features (provide statistics)

**Collate Function**:
```python
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Handles variable-length sequences with padding.

    Returns:
        {
            "inputs": (batch, max_seq_len, 28),
            "targets": (batch, max_seq_len, 4),
            "lengths": (batch,),
            "mask": (batch, max_seq_len),  # 1 for real, 0 for padding
            "metadata": {
                "participants": List[str],
                "trial_names": List[str],
            }
        }
    """
```

**Splitting Strategy**:
```python
# Leave-one-subject-out (LOSO)
train_participants = ["BT01", "BT02", ..., "BT13"]
val_participants = ["BT14"]
test_participants = ["BT15"]

train_dataset = ExoskeletonDataset(participants=train_participants)
val_dataset = ExoskeletonDataset(participants=val_participants)
```

**Error Handling**:
- Gracefully handle missing cache → download from HF
- Validate preprocessed cache integrity
- Provide helpful error messages for common issues

---

### Task 4: Data Download Utility

**File**: `src/exoskeleton_ml/data/download.py`

**Function**: `download_and_cache_dataset()`

**Responsibilities**:
1. Check if data already cached locally
2. If not, download from HuggingFace
3. Show download progress
4. Validate downloaded data
5. Trigger preprocessing and caching

**Implementation**:
```python
from datasets import load_dataset
from pathlib import Path
import json

def download_and_cache_dataset(
    hf_repo: str,
    cache_dir: Path,
    force_redownload: bool = False,
) -> Dataset:
    """
    Download dataset from HuggingFace and cache locally.

    Args:
        hf_repo: HuggingFace repository ID
        cache_dir: Local cache directory
        force_redownload: Force re-download even if cached

    Returns:
        HuggingFace Dataset object
    """
    metadata_file = cache_dir / "metadata.json"

    # Check if already cached
    if metadata_file.exists() and not force_redownload:
        print(f"✅ Dataset already cached at {cache_dir}")
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"   Participants: {metadata['participants']}")
        print(f"   Total trials: {metadata['total_trials']}")
        return load_from_disk(str(cache_dir / "hf_cache"))

    # Download from HuggingFace
    print(f"⬇️  Downloading dataset from {hf_repo}...")
    dataset = load_dataset(hf_repo)

    # Cache HF dataset
    hf_cache_dir = cache_dir / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(hf_cache_dir))

    # Save metadata
    participants = dataset.unique("participant")
    metadata = {
        "hf_repo": hf_repo,
        "participants": participants,
        "total_trials": len(dataset),
        "download_date": datetime.now().isoformat(),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Downloaded {len(dataset)} trials from {len(participants)} participants")
    return dataset
```

---

### Task 5: Integration with Training

**File**: `scripts/train.py` (update existing)

**Changes Required**:
1. Replace any existing data loading with new `ExoskeletonDataset`
2. Handle auto-download on first run
3. Support participant-based splits
4. Add data validation before training

**Example Usage**:
```python
from exoskeleton_ml.data import ExoskeletonDataset
from torch.utils.data import DataLoader

# Define splits (leave-one-subject-out)
train_participants = ["BT01", "BT02", "BT03", "BT06", "BT07",
                      "BT08", "BT09", "BT10", "BT11", "BT12", "BT13"]
val_participants = ["BT14"]
test_participants = ["BT15"]

# Create datasets (auto-downloads on first use)
train_dataset = ExoskeletonDataset(
    participants=train_participants,
    cache_dir="data/processed/phase1",
)

val_dataset = ExoskeletonDataset(
    participants=val_participants,
    cache_dir="data/processed/phase1",
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=train_dataset.collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=val_dataset.collate_fn,
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch["inputs"]      # (32, seq_len, 28)
        targets = batch["targets"]    # (32, seq_len, 4)
        mask = batch["mask"]          # (32, seq_len)

        # Your training logic here
        ...
```

---

### Task 6: Configuration Management

**File**: `configs/data/phase1.yaml`

**Purpose**: Centralize data configuration

**Contents**:
```yaml
# HuggingFace dataset configuration
hf_repo: macexo/exoskeleton-phase1
cache_dir: data/processed/phase1

# Feature configuration
num_imu_features: 24
num_angle_features: 4
num_targets: 4
total_input_features: 28

# Participant splits
splits:
  train: [BT01, BT02, BT03, BT06, BT07, BT08, BT09, BT10, BT11, BT12, BT13]
  val: [BT14]
  test: [BT15, BT16, BT17]

# DataLoader settings
batch_size: 32
num_workers: 4
shuffle_train: true
pin_memory: true

# Preprocessing options
normalize: true  # Whether to normalize features
normalization_stats_file: data/processed/phase1/normalization_stats.json
```

**Integration with Hydra**:
```python
from hydra import compose, initialize
from omegaconf import OmegaConf

@hydra.main(config_path="../configs", config_name="config")
def train(cfg):
    # Access data config
    dataset = ExoskeletonDataset(
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        participants=cfg.data.splits.train,
    )
```

---

### Task 7: Documentation

**File**: `docs/data_usage.md`

**Contents**:
1. How to download dataset for first time
2. How to use dataset in training
3. How to implement custom splits
4. How to add new data/participants
5. Troubleshooting common issues

**Example Structure**:
```markdown
# Data Usage Guide

## First-Time Setup

### 1. Download Data
The first time you run training, the dataset will auto-download:
```bash
python scripts/train.py
# Will download ~10GB from HuggingFace (one-time)
```

### 2. Manual Download (Optional)
```python
from exoskeleton_ml.data import download_and_cache_dataset
download_and_cache_dataset("macexo/exoskeleton-phase1", "data/processed/phase1")
```

## Using the Dataset

### Basic Usage
[code examples]

### Custom Splits
[code examples]

### Data Augmentation
[future section]

## Troubleshooting

**Problem**: Dataset download fails
**Solution**: Check internet connection, HF token, disk space

**Problem**: "Dataset not found" error
**Solution**: [steps]
```

---

## Testing & Validation

### Test 1: Data Preparation
```bash
# Run preparation script
python scripts/prepare_hf_dataset.py --validate

# Expected output:
# - ~1900 trials processed successfully
# - 15 participants
# - No critical errors
# - Dataset size ~5-10GB (compressed Parquet)

# Validation checks:
# ✓ All participants have expected number of trials
# ✓ No NaN/inf values in any trial
# ✓ All sequence lengths > 0
# ✓ All features present
```

### Test 2: HuggingFace Upload & Download
```bash
# Upload (requires HF token)
python scripts/upload_to_hf.py --repo macexo/exoskeleton-phase1

# Download on different machine
python -c "from datasets import load_dataset; \
           ds = load_dataset('macexo/exoskeleton-phase1'); \
           print(f'Downloaded {len(ds)} trials')"

# Expected: Successfully downloads and prints trial count
```

### Test 3: PyTorch Dataset
```python
from exoskeleton_ml.data import ExoskeletonDataset
from torch.utils.data import DataLoader

# Test loading
dataset = ExoskeletonDataset(participants=["BT01"])
assert len(dataset) > 0, "Dataset is empty"

# Test iteration
loader = DataLoader(dataset, batch_size=4)
batch = next(iter(loader))
assert batch["inputs"].shape[0] == 4, "Batch size mismatch"
assert batch["inputs"].shape[2] == 28, "Feature count mismatch"
assert batch["targets"].shape[2] == 4, "Target count mismatch"

print("✅ All dataset tests passed")
```

### Test 4: End-to-End Training
```bash
# Run training for 1 epoch with small subset
python scripts/train.py \
    data.splits.train=[BT01] \
    data.splits.val=[BT02] \
    training.num_epochs=1 \
    training.batch_size=8

# Expected:
# - Auto-downloads data if not cached
# - Trains successfully for 1 epoch
# - No errors or crashes
```

### Test 5: Data Integrity
```python
from exoskeleton_ml.data.validate import validate_dataset

# Check for common issues
issues = validate_dataset("data/hf_dataset")

# Expected issues to check:
# - Missing trials
# - Duplicate trials
# - Invalid participant IDs
# - Mismatched sequence lengths
# - NaN/inf values
# - Feature count mismatches

assert len(issues) == 0, f"Data validation failed: {issues}"
```

---

## Usage Patterns

### Pattern 1: Standard Training
```python
# Just run training - data downloads automatically
python scripts/train.py
```

### Pattern 2: Custom Split (Cross-Validation)
```python
# K-fold by participants
participants = ["BT01", ..., "BT15"]
for fold in range(5):
    test_participants = participants[fold*3:(fold+1)*3]
    train_participants = [p for p in participants if p not in test_participants]

    train_dataset = ExoskeletonDataset(participants=train_participants)
    test_dataset = ExoskeletonDataset(participants=test_participants)
    # Train model...
```

### Pattern 3: Data Exploration
```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("macexo/exoskeleton-phase1")

# Explore
print(f"Total trials: {len(dataset)}")
print(f"Participants: {dataset.unique('participant')}")

# Filter by task type
walking_trials = dataset.filter(lambda x: 'walk' in x['trial_name'])
print(f"Walking trials: {len(walking_trials)}")

# Get specific trial
trial = dataset[0]
print(f"Participant: {trial['participant']}")
print(f"Trial: {trial['trial_name']}")
print(f"Sequence length: {trial['sequence_length']}")
```

### Pattern 4: Updating Dataset
```python
# When new participants are added
# 1. Add new data to data/Phase1/BT18/...
# 2. Re-run preparation
python scripts/prepare_hf_dataset.py

# 3. Upload updated version
python scripts/upload_to_hf.py

# 4. On other machines, force re-download
dataset = ExoskeletonDataset(force_redownload=True)
```

---

## Future Considerations

### Short-term Improvements
1. **Data Augmentation**: Time warping, noise injection for robustness
2. **Streaming**: For very large datasets, stream from HF without full download
3. **Compression**: Experiment with quantization for smaller storage
4. **Validation Metrics**: Compute and store dataset statistics (mean, std, min, max)

### Medium-term
1. **Multi-phase Support**: Add Phase2, Phase3 datasets when available
2. **Task Classification**: Group trials by task type for task-specific models
3. **Synchronized Loading**: Multi-modal data (video, force plates) if available
4. **Online Learning**: Support for incremental dataset updates

### Long-term
1. **Real-time Streaming**: For live exoskeleton data collection
2. **Federated Learning**: Privacy-preserving training across institutions
3. **Data Versioning**: Track which model trained on which data version
4. **Automatic Splits**: ML-based intelligent train/val/test splitting

---

## File Structure After Implementation

```
exoskeleton-ai/
├── configs/
│   └── data/
│       └── phase1.yaml                    # Data configuration
├── data/
│   ├── Phase1/                            # Original CSV data (42GB)
│   ├── hf_dataset/                        # Prepared HF dataset (Parquet)
│   └── processed/
│       └── phase1/                        # Preprocessed cache
│           ├── metadata.json
│           ├── hf_cache/                  # HF download cache
│           ├── BT01/                      # Preprocessed tensors
│           ├── BT02/
│           └── ...
├── docs/
│   ├── data_infrastructure_plan.md        # This document
│   └── data_usage.md                      # User guide
├── scripts/
│   ├── prepare_hf_dataset.py              # Step 1: CSV → HF Dataset
│   ├── upload_to_hf.py                    # Step 2: Upload to HF Hub
│   └── train.py                           # Updated training script
└── src/exoskeleton_ml/data/
    ├── __init__.py
    ├── datasets.py                        # ExoskeletonDataset class
    ├── download.py                        # Download utilities
    ├── preprocessing.py                   # Preprocessing functions
    └── validate.py                        # Validation utilities
```

---

## Acceptance Criteria

### Phase 1: Preparation ✓
- [ ] `prepare_hf_dataset.py` successfully processes all Phase1 data
- [ ] No critical errors, < 1% trials skipped
- [ ] Dataset size reasonable (5-15GB Parquet)
- [ ] All 15 participants included
- [ ] Validation checks pass

### Phase 2: Upload ✓
- [ ] Dataset uploaded to HuggingFace Hub
- [ ] Dataset card complete and accurate
- [ ] Can download on different machine
- [ ] Download completes in < 30 min on good connection

### Phase 3: PyTorch Integration ✓
- [ ] `ExoskeletonDataset` class implemented
- [ ] Auto-download works on first use
- [ ] Local caching works (no re-download on subsequent runs)
- [ ] DataLoader integration works
- [ ] Collate function handles variable-length sequences

### Phase 4: Training Integration ✓
- [ ] `scripts/train.py` uses new dataset
- [ ] Training runs successfully end-to-end
- [ ] Splits by participant work correctly
- [ ] Performance acceptable (data loading not bottleneck)

### Phase 5: Documentation ✓
- [ ] Implementation plan complete (this document)
- [ ] User guide written (`docs/data_usage.md`)
- [ ] Code comments and docstrings complete
- [ ] Example usage patterns documented

---

## Success Metrics

1. **Download Time**: < 30 minutes on standard connection
2. **First Training Start**: < 5 minutes from clone to first epoch
3. **Data Loading Speed**: Not a bottleneck (< 10% of epoch time)
4. **Storage Efficiency**: < 15GB preprocessed cache
5. **Error Rate**: < 1% trials skipped during preparation
6. **Team Adoption**: All team members successfully use within 1 week

---

## Questions & Decisions

### Resolved
- ✅ Use HuggingFace (not DVC)
- ✅ Convert to Parquet (automatic via HF)
- ✅ No train/val/test split in HF repo (done in code)
- ✅ Required files: exo.csv, angle_filt.csv, moment_filt_bio.csv
- ✅ 28 input features, 4 output targets

### Pending
- [ ] HuggingFace repo name: `macexo/exoskeleton-phase1` or different?
- [ ] Public or private repository?
- [ ] Normalization strategy: global stats vs per-participant?
- [ ] Sequence length handling: pad to max, or dynamic batching?
- [ ] Data augmentation: when and what type?

---

## Contact & Support

- **Data Questions**: [Your team channel]
- **HuggingFace Issues**: Check `datasets` library docs
- **Implementation Help**: Refer to this plan or create GitHub issue

---

**End of Implementation Plan**
