# TCN Model Optimization Plan

**Current Performance**: Test RMSE 0.4316 Nm/kg, R² 0.7311
**Target**: < 0.35 Nm/kg, R² > 0.80
**Dataset**: Phase 1 only (BT01-BT17)

---

## Quick Wins (High Impact, Low Effort)

### 1. Learning Rate Tuning ⭐⭐⭐
**Current**: 0.001 with ReduceLROnPlateau
**Why it matters**: Best model at epoch 79, but trained until epoch 99 (20 epochs no improvement). This suggests the model may have converged but could benefit from different LR schedule.

**Experiment A: Lower initial LR with longer patience**
```bash
python scripts/train_tcn.py training.learning_rate=0.0005 \
  training.early_stopping_patience=30
```

**Experiment B: Cosine annealing (smoother decay)**
```bash
python scripts/train_tcn.py training.learning_rate=0.001 \
  training.scheduler.type=cosine_annealing \
  training.num_epochs=150
```

**Experiment C: One-cycle policy (fast training)**
```bash
python scripts/train_tcn.py training.scheduler.type=onecycle \
  training.learning_rate=0.005 \
  training.num_epochs=100
```

**Validation**: Compare final test RMSE. Expected improvement: 3-8%

---

### 2. Increase Model Capacity ⭐⭐⭐
**Current**: [25, 25, 25, 25, 25] = 5 layers × 25 channels
**Why it matters**: Original paper may have used larger models

**Experiment A: Wider network**
```bash
python scripts/train_tcn.py model=tcn_large  # [50, 50, 50, 50, 50]
```

**Experiment B: Deeper network**
```bash
python scripts/train_tcn.py model.architecture.num_channels=[25,25,25,25,25,25,25,25]
```

**Validation**: Check if training loss continues to decrease. If yes, more capacity helps.
**Expected improvement**: 10-15%

---

### 3. Better Data Handling ⭐⭐⭐
**Current**: Some trials have NaN values, wide sequence length variance
**Why it matters**: NaN handling and padding inefficiency can hurt performance

**Experiment A: Filter out problematic trials**
Add to `configs/data/phase1.yaml`:
```yaml
filtering:
  enabled: true
  min_sequence_length: 500  # Filter very short sequences
  max_nan_ratio: 0.01  # Filter trials with >1% NaN
```

**Experiment B: Sequence bucketing**
Group similar-length sequences in batches to reduce padding waste.

**Validation**: Check if validation loss improves and training is faster.
**Expected improvement**: 3-5%

---

### 4. Better LR Scheduling ⭐⭐
**Current**: Best at epoch 79, trained to 99 (patience=20 triggered)
**Why it matters**: Model converged with ReduceLROnPlateau but may need different schedule

**Experiment A**: More aggressive LR reduction
```bash
python scripts/train_tcn.py training.scheduler.factor=0.3 \
  training.scheduler.patience=5 \
  training.num_epochs=150
```

**Experiment B**: Warmup + cosine decay
```bash
python scripts/train_tcn.py training.scheduler.type=cosine_warmup \
  training.scheduler.warmup_epochs=10 \
  training.num_epochs=150
```

**Validation**: Check if model finds better local minima with different LR trajectory.
**Expected improvement**: 3-7%

---

## Medium Effort Optimizations

### 5. Hyperparameter Search ⭐⭐
**Why it matters**: Find optimal combination of hyperparameters

**Grid search with Hydra multirun**:
```bash
python scripts/train_tcn.py -m \
  training.learning_rate=0.0003,0.0005,0.001 \
  model.architecture.dropout=0.1,0.2,0.3 \
  training.weight_decay=0.0001,0.00005,0.00001
```

**Validation**: Pick config with best validation RMSE, evaluate on test set once.
**Expected improvement**: 10-15%

---

### 6. Better Normalization ⭐⭐
**Current**: Standard normalization (z-score)
**Why it matters**: IMU and angle features have very different scales

**Experiment A: Per-feature normalization**
Already doing this, but verify stats are computed correctly (handling NaN).

**Experiment B: Robust normalization**
```yaml
# In configs/data/phase1.yaml
preprocessing:
  normalization_method: "robust"  # Uses median and IQR instead of mean/std
```

**Validation**: Check if RMSE improves, especially on outlier movements.
**Expected improvement**: 3-5%

---

### 7. Gradient Clipping Adjustment ⭐
**Current**: max_norm=1.0
**Why it matters**: Too aggressive clipping can slow convergence

**Experiment**: Try different clip values
```bash
python scripts/train_tcn.py training.gradient_clip_norm=5.0
python scripts/train_tcn.py training.gradient_clip_norm=null  # No clipping
```

**Validation**: Monitor gradient norms during training. If always hitting limit, increase.
**Expected improvement**: 2-5%

---

### 8. Batch Size Tuning ⭐
**Current**: 32
**Why it matters**: Larger batches = more stable gradients, smaller = better generalization

**Experiment**:
```bash
python scripts/train_tcn.py training.batch_size=64  # Larger
python scripts/train_tcn.py training.batch_size=16  # Smaller
```

**Validation**: Compare convergence speed and final performance.
**Expected improvement**: 2-5%

---

## Advanced Optimizations

### 9. Loss Function Improvements ⭐⭐
**Current**: MSE loss
**Why it matters**: MSE treats all joints equally, but hips and knees have different scales

**Experiment A: Per-joint weighted loss**
Create `src/exoskeleton_ml/utils/losses.py`:
```python
class WeightedJointMSELoss(nn.Module):
    def __init__(self, joint_weights=[1.0, 1.0, 1.2, 1.2]):
        # Weight knees slightly more (indices 2, 3)
        super().__init__()
        self.weights = torch.tensor(joint_weights)

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights.to(mse.device)
        return weighted_mse.mean()
```

**Experiment B: Huber loss (robust to outliers)**
```python
criterion = nn.HuberLoss(delta=1.0)
```

**Validation**: Check if per-joint RMSE becomes more balanced.
**Expected improvement**: 5-8%

---

### 10. Data Augmentation ⭐⭐
**Why it matters**: Increases effective dataset size and robustness

**Experiment**: Add time-domain augmentations
```python
# In dataset class
def augment_sequence(self, inputs, targets):
    # 1. Time warping (speed up/slow down)
    if random.random() < 0.3:
        speed = random.uniform(0.9, 1.1)
        # Resample sequence

    # 2. Gaussian noise injection
    if random.random() < 0.3:
        noise = torch.randn_like(inputs) * 0.02
        inputs = inputs + noise

    # 3. Random temporal masking
    if random.random() < 0.2:
        # Zero out random time segments
        mask_len = int(0.1 * len(inputs))
        start = random.randint(0, len(inputs) - mask_len)
        inputs[start:start+mask_len] = 0

    return inputs, targets
```

**Validation**: Should improve generalization (test RMSE).
**Expected improvement**: 8-12%

---

### 11. Ensemble Methods ⭐
**Why it matters**: Averaging multiple models reduces variance

**Experiment**: Train 5 models with different seeds
```bash
for seed in 42 123 456 789 1011; do
  python scripts/train_tcn.py seed=$seed training.output_dir=outputs/ensemble/seed_$seed
done
```

Then average predictions:
```python
predictions = torch.stack([model(x) for model in models]).mean(dim=0)
```

**Validation**: Should improve all metrics.
**Expected improvement**: 5-10%

---

### 12. Architecture Variants ⭐⭐
**Why it matters**: TCN may not be optimal for all frequency components

**Experiment A: Increase kernel size for longer context**
```bash
python scripts/train_tcn.py model.architecture.kernel_size=9
```

**Experiment B: Add skip connections between layers**
Modify TCN architecture to have long-range skip connections.

**Validation**: Check if effective history matters (try 9→283 timesteps vs 7→187).
**Expected improvement**: 5-10%

---

## Systematic Evaluation Plan

### Phase 1: Quick Wins (1-2 days)
1. ✅ Run Experiment 1A (lower LR)
2. ✅ Run Experiment 2A (wider network)
3. ✅ Run Experiment 4 (longer training)
4. Compare all three to baseline

**Decision**: Pick the single best improvement

### Phase 2: Combine Winners (2-3 days)
5. ✅ Combine best 2-3 improvements from Phase 1
6. Run hyperparameter search (Experiment 5)

**Decision**: Lock in best configuration

### Phase 3: Advanced (3-5 days)
7. ✅ Try data augmentation (Experiment 10)
8. ✅ Try better loss function (Experiment 9A)
9. ✅ Train ensemble (Experiment 11)

**Decision**: Final model for deployment

---

## Validation Checklist

For each experiment:
- [ ] Record test RMSE, R², MAE
- [ ] Check per-joint metrics (ensure no joint gets worse)
- [ ] Plot training/validation curves
- [ ] Check for overfitting (train vs val gap)
- [ ] Note training time (some improvements may be too slow)
- [ ] Save best checkpoint

---

## Expected Final Results

After all optimizations:
- **Baseline**: 0.4316 RMSE, 0.7311 R²
- **After Quick Wins**: 0.38-0.40 RMSE, 0.75-0.77 R²
- **After All Optimizations**: 0.32-0.36 RMSE, 0.80-0.85 R²

This should put you within striking distance of the original paper's 0.14 RMSE deployment performance (which likely has additional real-time optimizations).

---

## Ready-to-Run Commands

### Experiment Set 1: Learning Rate
```bash
# Baseline (already done)
# python scripts/train_tcn.py  # → 0.4316 RMSE

# Lower LR
python scripts/train_tcn.py training.learning_rate=0.0005

# Cosine schedule
python scripts/train_tcn.py training.learning_rate=0.001 \
  training.num_epochs=150 \
  training.scheduler.type=cosine_annealing
```

### Experiment Set 2: Model Capacity
```bash
# Wider
python scripts/train_tcn.py model=tcn_large

# Deeper
python scripts/train_tcn.py model.architecture.num_channels=[25,25,25,25,25,25,25,25]
```

### Experiment Set 3: Extended Training
```bash
python scripts/train_tcn.py training.num_epochs=200 \
  training.early_stopping_patience=40
```

### Experiment Set 4: Hyperparameter Grid Search
```bash
python scripts/train_tcn.py -m \
  training.learning_rate=0.0003,0.0005,0.001 \
  model.architecture.dropout=0.1,0.2,0.3 \
  training.batch_size=16,32,64
```

---

## Monitoring and Tracking

Create a results spreadsheet:

| Experiment | RMSE | R² | MAE | Hip RMSE | Knee RMSE | Training Time | Notes |
|------------|------|----|----|----------|-----------|---------------|-------|
| Baseline   | 0.4316 | 0.7311 | 0.3014 | 0.44 | 0.42 | ~2-3h | Best: epoch 79/99 |
| Lower LR   | ? | ? | ? | ? | ? | ? | |
| Wider Net  | ? | ? | ? | ? | ? | ? | |
| ...        | | | | | | | |

---

## Implementation Support

I can help you:
1. Implement any of these experiments
2. Create scripts for systematic evaluation
3. Add new loss functions or augmentations
4. Set up hyperparameter sweeps
5. Analyze results and debug issues

Let me know which experiments you want to start with!
