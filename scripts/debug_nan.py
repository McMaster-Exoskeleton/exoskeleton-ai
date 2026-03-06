"""Debug script to check for NaN values in data and model outputs."""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exoskeleton_ml.data import create_dataloaders
from exoskeleton_ml.models import create_model
from exoskeleton_ml.utils import get_device


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def debug_nan(cfg: DictConfig) -> None:
    """Debug NaN issues in data and model."""
    print("=" * 80)
    print("NaN Debugging Script")
    print("=" * 80)

    # Setup device
    device = get_device()
    print(f"\nDevice: {device}")

    # Create dataloaders
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        train_participants=cfg.data.splits.train,
        val_participants=cfg.data.splits.val,
        test_participants=cfg.data.splits.test,
        batch_size=cfg.training.batch_size,
        num_workers=0,  # Single process for debugging
        normalize=cfg.data.preprocessing.get("normalize", True),
    )

    # Check first batch of training data
    print("\n2. Checking training data for NaN values...")
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["inputs"]
        targets = batch["targets"]
        mask = batch["mask"]

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Mask shape: {mask.shape}")

        # Check for NaN
        has_nan_inputs = torch.isnan(inputs).any()
        has_nan_targets = torch.isnan(targets).any()
        has_inf_inputs = torch.isinf(inputs).any()
        has_inf_targets = torch.isinf(targets).any()

        print(f"  NaN in inputs: {has_nan_inputs}")
        print(f"  NaN in targets: {has_nan_targets}")
        print(f"  Inf in inputs: {has_inf_inputs}")
        print(f"  Inf in targets: {has_inf_targets}")

        if has_nan_inputs:
            nan_count = torch.isnan(inputs).sum().item()
            print(f"  NaN count in inputs: {nan_count}/{inputs.numel()}")
            # Find which features have NaN
            nan_per_feature = torch.isnan(inputs).sum(dim=(0, 1))
            for i, count in enumerate(nan_per_feature):
                if count > 0:
                    print(f"    Feature {i}: {count} NaN values")

        if has_nan_targets:
            nan_count = torch.isnan(targets).sum().item()
            print(f"  NaN count in targets: {nan_count}/{targets.numel()}")

        # Print statistics
        print(f"  Input stats:")
        print(f"    Mean: {inputs[~torch.isnan(inputs)].mean():.4f}")
        print(f"    Std: {inputs[~torch.isnan(inputs)].std():.4f}")
        print(f"    Min: {inputs[~torch.isnan(inputs)].min():.4f}")
        print(f"    Max: {inputs[~torch.isnan(inputs)].max():.4f}")

        print(f"  Target stats:")
        print(f"    Mean: {targets[~torch.isnan(targets)].mean():.4f}")
        print(f"    Std: {targets[~torch.isnan(targets)].std():.4f}")
        print(f"    Min: {targets[~torch.isnan(targets)].min():.4f}")
        print(f"    Max: {targets[~torch.isnan(targets)].max():.4f}")

        if batch_idx >= 2:  # Check first 3 batches
            break

    # Create model
    print("\n3. Creating model...")
    model = create_model(cfg.model).to(device)
    print(f"  Model: {cfg.model.name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n4. Testing forward pass...")
    model.eval()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)

        print(f"\nBatch {batch_idx + 1}:")

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)

        print(f"  Output shape: {outputs.shape}")
        print(f"  NaN in outputs: {torch.isnan(outputs).any()}")
        print(f"  Inf in outputs: {torch.isinf(outputs).any()}")

        if torch.isnan(outputs).any():
            nan_count = torch.isnan(outputs).sum().item()
            print(f"  NaN count: {nan_count}/{outputs.numel()}")

        # Print output statistics
        if not torch.isnan(outputs).all():
            print(f"  Output stats:")
            print(f"    Mean: {outputs[~torch.isnan(outputs)].mean():.4f}")
            print(f"    Std: {outputs[~torch.isnan(outputs)].std():.4f}")
            print(f"    Min: {outputs[~torch.isnan(outputs)].min():.4f}")
            print(f"    Max: {outputs[~torch.isnan(outputs)].max():.4f}")

        if batch_idx >= 2:
            break

    # Test with gradient computation
    print("\n5. Testing backward pass...")
    model.train()
    criterion = torch.nn.MSELoss()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        print(f"\nBatch {batch_idx + 1}:")

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        mask_expanded = mask.unsqueeze(-1)
        masked_outputs = outputs * mask_expanded
        masked_targets = targets * mask_expanded

        loss = criterion(masked_outputs, masked_targets)

        print(f"  Loss: {loss.item()}")
        print(f"  Loss is NaN: {torch.isnan(loss)}")

        # Check gradients
        loss.backward()

        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in: {name}")
                    has_nan_grad = True

        if has_nan_grad:
            print("  ❌ Found NaN gradients!")
        else:
            print("  ✅ No NaN gradients")

        # Zero gradients for next iteration
        model.zero_grad()

        if batch_idx >= 2:
            break

    print("\n" + "=" * 80)
    print("Debug complete")
    print("=" * 80)


if __name__ == "__main__":
    debug_nan()
