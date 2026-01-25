# TCN Model Implementation Plan

**Project**: Exoskeleton AI - TCN Model for Joint Moment Estimation
**Goal**: Implement Temporal Convolutional Network (TCN) for task-agnostic biological joint moment estimation
**Date**: 2026-01-06
**Status**: Planning → Implementation

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Tasks](#implementation-tasks)
4. [Training Strategy](#training-strategy)
5. [Evaluation & Metrics](#evaluation--metrics)
6. [Configuration](#configuration)

---

## Overview

### Context
- Data pipeline is complete and tested (Phase1 dataset on HuggingFace)
- Reference TCN implementation available from original paper
- Need to adapt TCN for our specific use case: 28 input features → 4 output targets

### Goals
1. **Implement TCN Model**: Adapt reference implementation for our data format
2. **Configure Architecture**: Set hyperparameters based on paper and experiments
3. **Training Script**: End-to-end training loop with validation
4. **Evaluation**: Metrics for regression performance (RMSE, MAE, R²)
5. **Checkpointing**: Save/load model weights for inference

### Reference Paper
- **Title**: "Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation"
- **Original Code**: `temp/capsule-5421243/code/tcn.py`
- **Key Innovation**: TCN predicts biological joint moments from IMU + joint angle data

---

## Architecture

### Model Overview

```
Input: (batch, seq_len, 28)
  ├─ 24 IMU features (4 IMUs × 6 DOF)
  └─ 4 joint angles (hip/knee left/right)
       ↓
  [Normalization Layer]
       ↓
  [Temporal Convolutional Network]
    - Multiple residual blocks
    - Dilated causal convolutions
    - Exponentially increasing receptive field
       ↓
  [Linear Output Layer]
       ↓
Output: (batch, seq_len, 4)
  └─ 4 joint moments (hip/knee left/right, Nm/kg)
```

### TCN Components

#### 1. **Temporal Block** (Building Block)
```
Input → [Conv1d] → [Norm] → [Chomp] → [ReLU] → [Dropout]
     → [Conv1d] → [Norm] → [Chomp] → [ReLU] → [Dropout]
     → Add Residual Connection → [ReLU] → Output
```

**Key Features**:
- **Causal Convolutions**: No future information leakage
- **Dilations**: Exponentially increasing (1, 2, 4, 8, ...)
- **Residual Connections**: Skip connections for gradient flow
- **Weight Normalization**: Stabilizes training

#### 2. **Temporal Convolutional Network** (Stacked Blocks)
- Multiple temporal blocks stacked sequentially
- Each block increases receptive field by 2^i
- Final receptive field (effective history): `(kernel_size - 1) × 2^(num_levels) + 1`

#### 3. **Output Layer**
- Linear layer: `num_channels[-1] → 4`
- No activation (regression task)

### Architecture Parameters

Based on original paper and our data:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `input_size` | 28 | 24 IMU + 4 angles |
| `output_size` | 4 | 4 joint moments |
| `num_channels` | [25, 25, 25, 25, 25] | 5 layers × 25 channels |
| `kernel_size` | 7 | Receptive field balance |
| `dropout` | 0.2 | Regularization |
| `spatial_dropout` | False | Standard dropout |
| `activation` | ReLU | Non-linearity |
| `norm` | weight_norm | Weight normalization |

**Effective History Calculation**:
```
eff_hist = (kernel_size - 1) × (2^num_levels - 1) + 1
         = (7 - 1) × (2^5 - 1) + 1
         = 6 × 31 + 1
         = 187 timesteps
```

At 100Hz sampling, this is **1.87 seconds** of historical context.

---

## Implementation Tasks

### Task 1: TCN Model Implementation

**File**: `src/exoskeleton_ml/models/tcn.py`

**Components to Implement**:

#### 1.1 `Chomp1d` Module
```python
class Chomp1d(nn.Module):
    """Removes trailing padding to ensure causal convolutions."""
    def __init__(self, chomp_size: int):
        # Remove last chomp_size elements from sequence
```

#### 1.2 `TemporalBlock` Module
```python
class TemporalBlock(nn.Module):
    """Single residual block with dilated convolutions."""
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        dropout_type: str = 'Dropout',
        activation: str = 'ReLU',
        norm: str = 'weight_norm',
    ):
        # Two conv layers with residual connection
```

#### 1.3 `TemporalConvNet` Module
```python
class TemporalConvNet(nn.Module):
    """Stacks temporal blocks with increasing dilation."""
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        dropout_type: str = 'Dropout',
        activation: str = 'ReLU',
        norm: str = 'weight_norm',
    ):
        # Create num_levels blocks with dilation 2^i
```

#### 1.4 `TCN` Module (Main Model)
```python
class TCN(nn.Module):
    """Complete TCN model for joint moment estimation."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int,
        dropout: float,
        eff_hist: int,
        spatial_dropout: bool = False,
        activation: str = 'ReLU',
        norm: str = 'weight_norm',
        center: Optional[torch.Tensor] = None,  # Normalization mean
        scale: Optional[torch.Tensor] = None,   # Normalization std
    ):
```

**Key Modifications from Reference**:
1. **Input Format**: Expect (batch, seq_len, features) instead of (batch, features, seq_len)
   - Add transpose before TCN
   - Transpose back after linear layer
2. **Optional Normalization**: Support external normalization (from dataset)
3. **Better Type Hints**: Add proper typing for clarity
4. **Configurable via Hydra**: All params from config file

---

### Task 2: Model Configuration

**File**: `configs/model/tcn.yaml`

```yaml
# TCN Model Configuration
# Based on "Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation"

name: tcn
type: tcn

# Architecture
architecture:
  input_size: 28  # 24 IMU + 4 angles
  output_size: 4  # 4 joint moments
  num_channels: [25, 25, 25, 25, 25]  # 5 layers, 25 filters each
  kernel_size: 7
  dropout: 0.2
  spatial_dropout: false
  activation: ReLU
  norm: weight_norm

# Computed properties
effective_history: 187  # timesteps (1.87s at 100Hz)
receptive_field_ms: 1870  # milliseconds

# Normalization (handled by dataset, but model can do it too)
normalization:
  enabled: false  # Dataset already normalizes
  method: standard  # Options: standard, minmax
  learn_stats: false  # Use dataset stats, not learnable

# Weight initialization
initialization:
  conv_std: 0.01  # Conv layer init std
  linear_std: 0.01  # Linear layer init std

# Model size estimation
estimated_params: ~50000  # Approximate parameter count
```

**Variants** (for experimentation):

Create additional configs for architecture search:
- `configs/model/tcn_small.yaml`: [15, 15, 15, 15] (smaller, faster)
- `configs/model/tcn_large.yaml`: [50, 50, 50, 50, 50, 50] (larger, more capacity)
- `configs/model/tcn_deep.yaml`: [25] × 8 layers (deeper network)

---

### Task 3: Model Factory Update

**File**: `src/exoskeleton_ml/models/model_factory.py`

**Current State**: Basic factory with baseline model

**Updates Needed**:
```python
from typing import Any, Dict
import torch.nn as nn
from omegaconf import DictConfig

from .baseline import BaselineModel
from .tcn import TCN

def create_model(config: DictConfig) -> nn.Module:
    """Create model from config.

    Args:
        config: Model configuration (from configs/model/*.yaml)

    Returns:
        Initialized PyTorch model
    """
    model_type = config.type

    if model_type == "baseline":
        return BaselineModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif model_type == "tcn":
        # Calculate effective history
        num_levels = len(config.architecture.num_channels)
        eff_hist = (config.architecture.kernel_size - 1) * (2 ** num_levels - 1) + 1

        return TCN(
            input_size=config.architecture.input_size,
            output_size=config.architecture.output_size,
            num_channels=config.architecture.num_channels,
            kernel_size=config.architecture.kernel_size,
            dropout=config.architecture.dropout,
            eff_hist=eff_hist,
            spatial_dropout=config.architecture.spatial_dropout,
            activation=config.architecture.activation,
            norm=config.architecture.norm,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

---

### Task 4: Training Script

**File**: `scripts/train.py`

**Requirements**:
1. Load data using `ExoskeletonDataset` and `create_dataloaders`
2. Initialize TCN model from config
3. Training loop with:
   - Loss function: MSE (Mean Squared Error) for regression
   - Optimizer: Adam with configurable LR
   - Learning rate scheduler: ReduceLROnPlateau or CosineAnnealing
   - Gradient clipping: Prevent exploding gradients
4. Validation loop every N epochs
5. Checkpointing: Save best model based on val loss
6. Logging: TensorBoard or Weights & Biases
7. Early stopping: Stop if no improvement for M epochs

**Structure**:
```python
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from exoskeleton_ml.data import create_dataloaders
from exoskeleton_ml.models import create_model
from exoskeleton_ml.utils import EarlyStopping, save_checkpoint, load_checkpoint


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Main training function."""

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        train_participants=cfg.data.splits.train,
        val_participants=cfg.data.splits.val,
        test_participants=cfg.data.splits.test,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        normalize=cfg.data.preprocessing.normalize,
    )

    # 3. Create model
    model = create_model(cfg.model).to(device)

    # 4. Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 5. Training loop
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=cfg.training.early_stopping_patience)

    for epoch in range(cfg.training.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, output_dir / "best_model.pt")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # 6. Final evaluation on test set
    test_loss, test_metrics = evaluate_test_set(model, test_loader, criterion, device)

    print(f"Final Test Loss: {test_loss:.4f}")


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs = batch['inputs'].to(device)  # (batch, seq_len, 28)
        targets = batch['targets'].to(device)  # (batch, seq_len, 4)
        mask = batch['mask'].to(device)  # (batch, seq_len)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len, 4)

        # Compute loss (only on non-padded positions)
        loss = masked_loss(outputs, targets, mask, criterion)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)

            outputs = model(inputs)
            loss = masked_loss(outputs, targets, mask, criterion)

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(outputs, targets, mask)

    return avg_loss, metrics


def masked_loss(outputs, targets, mask, criterion):
    """Compute loss only on non-padded positions."""
    # Expand mask to match output dimensions
    mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

    # Apply mask
    masked_outputs = outputs * mask
    masked_targets = targets * mask

    # Compute loss
    loss = criterion(masked_outputs, masked_targets)

    # Normalize by number of valid positions
    num_valid = mask.sum()
    loss = loss * mask.numel() / num_valid

    return loss


if __name__ == "__main__":
    train()
```

---

### Task 5: Evaluation Metrics

**File**: `src/exoskeleton_ml/utils/metrics.py`

**Metrics to Implement**:

1. **RMSE (Root Mean Squared Error)**
   - Primary metric for regression
   - Per-joint and overall

2. **MAE (Mean Absolute Error)**
   - More interpretable than RMSE
   - Less sensitive to outliers

3. **R² Score (Coefficient of Determination)**
   - Measures explained variance
   - Per-joint correlation

4. **Normalized RMSE (NRMSE)**
   - RMSE normalized by target range
   - Allows cross-joint comparison

5. **Per-Participant Metrics**
   - Evaluate generalization to unseen participants
   - Critical for leave-one-subject-out validation

```python
import torch
from typing import Dict, Tuple

def compute_metrics(
    predictions: torch.Tensor,  # (batch, seq_len, 4)
    targets: torch.Tensor,       # (batch, seq_len, 4)
    mask: torch.Tensor,          # (batch, seq_len)
) -> Dict[str, float]:
    """Compute regression metrics.

    Returns:
        Dictionary with metrics:
        - rmse_overall: Overall RMSE across all joints
        - rmse_hip_l, rmse_hip_r, rmse_knee_l, rmse_knee_r: Per-joint RMSE
        - mae_overall: Overall MAE
        - r2_overall: Overall R² score
        - nrmse_overall: Normalized RMSE
    """
    # Mask predictions and targets
    mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
    pred_masked = predictions * mask
    target_masked = targets * mask
    num_valid = mask.sum()

    # RMSE
    mse = ((pred_masked - target_masked) ** 2).sum() / num_valid
    rmse = torch.sqrt(mse).item()

    # Per-joint RMSE
    joint_names = ['hip_l', 'hip_r', 'knee_l', 'knee_r']
    per_joint_rmse = {}
    for i, joint in enumerate(joint_names):
        joint_mse = ((pred_masked[..., i] - target_masked[..., i]) ** 2).sum() / mask.sum()
        per_joint_rmse[f'rmse_{joint}'] = torch.sqrt(joint_mse).item()

    # MAE
    mae = (pred_masked - target_masked).abs().sum() / num_valid

    # R²
    ss_res = ((target_masked - pred_masked) ** 2).sum()
    ss_tot = ((target_masked - target_masked.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # NRMSE (normalized by range)
    target_range = target_masked.max() - target_masked.min()
    nrmse = rmse / target_range.item() if target_range.item() > 0 else 0.0

    return {
        'rmse_overall': rmse,
        **per_joint_rmse,
        'mae_overall': mae.item(),
        'r2_overall': r2.item(),
        'nrmse_overall': nrmse,
    }
```

---

### Task 6: Configuration Integration

**File**: `configs/config.yaml` (Update)

```yaml
# Main configuration file for exoskeleton ML project

defaults:
  - data: phase1
  - model: tcn
  - _self_

# Training configuration
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001

  # Optimizer
  optimizer: adam

  # Scheduler
  scheduler:
    type: reduce_on_plateau
    factor: 0.5
    patience: 10
    min_lr: 1.0e-6

  # Regularization
  gradient_clip_norm: 1.0
  early_stopping_patience: 20

  # Checkpointing
  checkpoint_dir: models/checkpoints
  save_frequency: 5  # Save every N epochs

  # Output
  output_dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Logging configuration
logging:
  backend: tensorboard  # Options: tensorboard, wandb, none
  log_dir: logs
  log_frequency: 10  # Log every N batches
  log_metrics: true
  log_gradients: false  # Can be expensive

  # Weights & Biases (if using)
  wandb:
    project: exoskeleton-ai
    entity: null  # Your W&B username/team
    tags: [tcn, phase1]

# Evaluation
evaluation:
  metrics:
    - rmse
    - mae
    - r2
    - nrmse
  per_joint: true
  per_participant: true
  save_predictions: false  # Save predictions for analysis

# Hardware
device: null  # Auto-detect (cuda, mps, or cpu)
seed: 42

# Hydra configuration
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: outputs/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## Training Strategy

### 1. Data Splitting

**Leave-One-Subject-Out (LOSO)**:
- Train: BT01-BT13 (11 participants)
- Validation: BT14 (1 participant)
- Test: BT15-BT17 (3 participants)

**Rationale**: Tests generalization to completely unseen participants

### 2. Hyperparameter Tuning

**Grid Search** (via Hydra multirun):
```bash
python scripts/train.py -m \
  model.architecture.num_channels=[15,15,15,15],[25,25,25,25,25],[50,50,50,50] \
  model.architecture.kernel_size=5,7,9 \
  training.learning_rate=0.0001,0.001,0.01
```

**Parameters to Tune**:
1. `num_channels`: Network width
2. `kernel_size`: Receptive field
3. `learning_rate`: Optimization speed
4. `dropout`: Regularization strength
5. `batch_size`: Memory vs convergence

### 3. Training Procedures

**Standard Training**:
```bash
python scripts/train.py
```

**Resume from Checkpoint**:
```bash
python scripts/train.py training.resume_from=models/checkpoints/best_model.pt
```

**Distributed Training** (multi-GPU):
```bash
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py
```

### 4. Monitoring

**TensorBoard**:
```bash
tensorboard --logdir logs
```

**Metrics to Track**:
- Training loss (per epoch)
- Validation loss (per epoch)
- Learning rate (per epoch)
- Per-joint RMSE (validation)
- Gradient norms (optional)

---

## Evaluation & Metrics

### Primary Metrics

1. **RMSE (Nm/kg)**: Lower is better
   - Expected range: 0.1-0.5 Nm/kg (based on paper)

2. **R² Score**: Higher is better (closer to 1.0)
   - Expected range: 0.7-0.95 (based on paper)

3. **Per-Joint Analysis**:
   - Hip moments typically higher magnitude than knee
   - Left/right symmetry expected for symmetric tasks

### Evaluation Protocol

1. **Validation Set** (BT14):
   - Used during training for model selection
   - Compute metrics every epoch

2. **Test Set** (BT15-BT17):
   - Final evaluation ONLY after training complete
   - Report mean ± std across participants

3. **Cross-Validation** (Optional):
   - 5-fold LOSO: Rotate which participants are test
   - More robust performance estimate

### Baseline Comparisons

Compare TCN against:
1. **Linear Regression**: Simple baseline
2. **LSTM**: Recurrent baseline
3. **1D CNN**: Non-dilated CNN
4. **Transformer**: Attention-based

---

## File Structure After Implementation

```
exoskeleton-ai/
├── configs/
│   ├── config.yaml                    # Main config (updated)
│   ├── data/
│   │   └── phase1.yaml                # Data config (existing)
│   └── model/
│       ├── baseline.yaml              # Baseline config (existing)
│       ├── tcn.yaml                   # TCN config (NEW)
│       ├── tcn_small.yaml             # TCN variant (NEW)
│       └── tcn_large.yaml             # TCN variant (NEW)
├── docs/
│   ├── data_infrastructure_plan.md    # Data plan (existing)
│   └── tcn_implementation_plan.md     # This document (NEW)
├── scripts/
│   ├── train.py                       # Training script (NEW)
│   ├── evaluate.py                    # Evaluation script (NEW)
│   └── test_dataloader.py             # Dataloader test (existing)
└── src/exoskeleton_ml/
    ├── data/
    │   ├── datasets.py                # Dataset class (existing)
    │   └── download.py                # Download utils (existing)
    ├── models/
    │   ├── __init__.py                # Model exports
    │   ├── baseline.py                # Baseline model (existing)
    │   ├── tcn.py                     # TCN model (NEW)
    │   └── model_factory.py           # Factory (UPDATE)
    └── utils/
        ├── __init__.py
        ├── metrics.py                 # Evaluation metrics (NEW)
        ├── checkpointing.py           # Save/load utils (NEW)
        └── early_stopping.py          # Early stopping (NEW)
```

---

## Implementation Checklist

### Phase 1: Model Implementation ✓
- [ ] Implement `Chomp1d` module
- [ ] Implement `TemporalBlock` module
- [ ] Implement `TemporalConvNet` module
- [ ] Implement `TCN` main model
- [ ] Add type hints and docstrings
- [ ] Write unit tests for each component

### Phase 2: Configuration ✓
- [ ] Create `configs/model/tcn.yaml`
- [ ] Update `configs/config.yaml`
- [ ] Create model variants (small, large)
- [ ] Update model factory to support TCN

### Phase 3: Training Infrastructure ✓
- [ ] Implement `scripts/train.py`
- [ ] Implement metrics in `utils/metrics.py`
- [ ] Implement checkpointing in `utils/checkpointing.py`
- [ ] Implement early stopping in `utils/early_stopping.py`
- [ ] Add logging (TensorBoard)

### Phase 4: Testing & Validation ✓
- [ ] Test model forward pass with dummy data
- [ ] Test training loop with small subset
- [ ] Test checkpointing save/load
- [ ] Test metric computation
- [ ] Verify gradient flow

### Phase 5: Full Training ✓
- [ ] Train on full dataset
- [ ] Monitor training curves
- [ ] Evaluate on validation set
- [ ] Hyperparameter tuning
- [ ] Final evaluation on test set

### Phase 6: Documentation ✓
- [ ] Document training procedure
- [ ] Document hyperparameters
- [ ] Document results
- [ ] Create model card
- [ ] Usage examples

---

## Expected Results

Based on original paper:

| Metric | Expected Value |
|--------|----------------|
| Overall RMSE | 0.2-0.4 Nm/kg |
| Hip RMSE | 0.3-0.5 Nm/kg |
| Knee RMSE | 0.15-0.3 Nm/kg |
| R² Score | 0.80-0.95 |
| Training Time | ~2-4 hours (GPU) |

**Note**: Our results may differ due to:
- Different data preprocessing
- Different participant splits
- Different training procedures

---

## Next Steps After Implementation

1. **Ablation Studies**:
   - Test without IMU data (angles only)
   - Test without angle data (IMU only)
   - Test different network depths

2. **Real-time Inference**:
   - Optimize for low-latency prediction
   - Export to ONNX for deployment
   - Test on embedded hardware

3. **Multi-phase Training**:
   - Add Phase2 and Phase3 datasets
   - Train on combined dataset
   - Test transfer learning

4. **Exoskeleton Integration**:
   - Real-time moment estimation
   - Control loop integration
   - Hardware testing

---

## Questions & Decisions

### Resolved
- ✅ Use TCN architecture from paper
- ✅ Input: 28 features (24 IMU + 4 angles)
- ✅ Output: 4 joint moments
- ✅ Loss function: MSE
- ✅ Data format: (batch, seq_len, features)

### Pending
- [ ] Final hyperparameters (will tune)
- [ ] Learning rate schedule (ReduceLROnPlateau vs Cosine)
- [ ] Batch size (limited by GPU memory)
- [ ] Data augmentation strategy
- [ ] Normalization: per-participant vs global?

---

**End of Implementation Plan**
