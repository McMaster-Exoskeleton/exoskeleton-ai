# TCN Model Quick Start Guide

This guide shows you how to train and use the TCN model for exoskeleton joint moment estimation.

## Prerequisites

✅ Data pipeline is set up (HuggingFace dataset available)
✅ Dependencies installed (`pip install -r requirements.txt`)
✅ Configuration files in place

## Quick Start

### 1. Test the Model (Optional but Recommended)

Verify that the TCN model works correctly:

```bash
python scripts/test_tcn_model.py
```

**Expected output**: All 5 tests should pass.

### 2. Train the Model

Train with default configuration (TCN medium, 100 epochs):

```bash
python scripts/train_tcn.py
```

This will:
- Download data from HuggingFace (if not cached)
- Create train/val/test dataloaders
- Initialize TCN model (~45K parameters)
- Train for up to 100 epochs with early stopping
- Save best model to `outputs/<timestamp>/best_model.pt`
- Evaluate on test set

**Training time**: ~2-4 hours on GPU, ~8-12 hours on CPU (for full dataset)

### 3. Monitor Training

Training progress is printed to console. Look for:

```
Epoch 10/100
--------------------------------------------------------------------------------
Train Loss: 0.1234
Val Loss:   0.1567
Val RMSE:   0.3456 Nm/kg
Val R²:     0.8234
Val MAE:    0.2789 Nm/kg
  Hip L:  0.3876 Nm/kg
  Hip R:  0.3912 Nm/kg
  Knee L: 0.2345 Nm/kg
  Knee R: 0.2289 Nm/kg
```

**Good signs**:
- Train and val loss both decreasing
- RMSE < 0.5 Nm/kg after several epochs
- R² > 0.7 after several epochs
- No huge gap between train and val loss (overfitting)

## Configuration Options

### Override Training Parameters

```bash
# Train for fewer epochs
python scripts/train_tcn.py training.num_epochs=50

# Use larger batch size (if you have GPU memory)
python scripts/train_tcn.py training.batch_size=64

# Change learning rate
python scripts/train_tcn.py training.learning_rate=0.0001

# Combine multiple overrides
python scripts/train_tcn.py training.num_epochs=50 training.batch_size=16 training.learning_rate=0.0005
```

### Use Different Model Sizes

```bash
# Small model (faster, fewer parameters)
python scripts/train_tcn.py model=tcn_small

# Large model (slower, more capacity)
python scripts/train_tcn.py model=tcn_large
```

### Custom Data Splits

Override participant splits in `configs/data/phase1.yaml` or via command line:

```bash
python scripts/train_tcn.py \
  data.splits.train=[BT01,BT02,BT03,BT06,BT07,BT08] \
  data.splits.val=[BT09] \
  data.splits.test=[BT10,BT11]
```

## Hyperparameter Tuning

Run a sweep over multiple hyperparameters:

```bash
python scripts/train_tcn.py -m \
  model=tcn_small,tcn,tcn_large \
  training.learning_rate=0.0001,0.001,0.01
```

This will train 9 models (3 sizes × 3 learning rates) in parallel runs.

## Output Files

After training, check `outputs/<timestamp>/`:

```
outputs/2026-01-06/14-30-00/
├── config.yaml                  # Full configuration used
├── best_model.pt                # Best model checkpoint
├── checkpoint_epoch_10.pt       # Periodic checkpoints
├── checkpoint_epoch_20.pt
├── training_results.pt          # Loss curves and metrics
└── .hydra/                      # Hydra logs
```

## Loading a Trained Model

```python
import torch
from exoskeleton_ml.models import TCN
from exoskeleton_ml.utils import load_checkpoint

# Create model
model = TCN(
    input_size=28,
    output_size=4,
    num_channels=[25, 25, 25, 25, 25],
    kernel_size=7,
    dropout=0.2,
    eff_hist=187,
)

# Load weights
checkpoint = load_checkpoint(
    "outputs/2026-01-06/14-30-00/best_model.pt",
    model,
    device="cpu",
)

# Use model for inference
model.eval()
with torch.no_grad():
    predictions = model(inputs)  # inputs shape: (batch, seq_len, 28)
```

## Expected Results

Based on the original paper, you should achieve:

| Metric | Target |
|--------|--------|
| Overall RMSE | 0.2-0.4 Nm/kg |
| Hip RMSE | 0.3-0.5 Nm/kg |
| Knee RMSE | 0.15-0.3 Nm/kg |
| R² Score | 0.80-0.95 |

Your results may vary based on:
- Data preprocessing
- Participant splits
- Hyperparameters
- Random initialization

## Troubleshooting

### Model not training (loss stays flat)

- **Check learning rate**: Try smaller (0.0001) or larger (0.01)
- **Check data**: Verify normalization is enabled in config
- **Check gradients**: Run test script to verify backward pass

### Out of memory errors

```bash
# Reduce batch size
python scripts/train_tcn.py training.batch_size=8

# Use smaller model
python scripts/train_tcn.py model=tcn_small

# Reduce number of workers
python scripts/train_tcn.py data.num_workers=0
```

### Data not downloading

- **Check HuggingFace access**: Verify you can access `MacExo/exoData`
- **Check disk space**: Need ~10GB free for cache
- **Manual download**: Run `scripts/test_dataloader.py` first

### Import errors

```bash
# Make sure you're in the project root
cd /path/to/exoskeleton-ai

# Verify Python path includes src/
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run from scripts directory
cd scripts
python train_tcn.py
```

## Next Steps

After successful training:

1. **Analyze Results**: Plot loss curves, per-joint performance
2. **Cross-Validation**: Train with different participant splits
3. **Ablation Studies**: Test without IMU or without angles
4. **Real-time Inference**: Export to ONNX, optimize for latency
5. **Hardware Testing**: Deploy on exoskeleton hardware

## Additional Resources

- **Implementation Plan**: `docs/tcn_implementation_plan.md`
- **Data Pipeline**: `docs/data_infrastructure_plan.md`
- **Model Tests**: `scripts/test_tcn_model.py`
- **Data Tests**: `scripts/test_dataloader.py`

## Support

If you encounter issues:
1. Check the implementation plan for detailed architecture info
2. Run test scripts to verify components work
3. Review config files for correct parameters
4. Check GitHub issues for known problems

---

**Happy Training! 🚀**
