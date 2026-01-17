"""
Training script for TCN model on exoskeleton joint moment estimation.

Usage:
    # Basic training with default config
    python scripts/train_tcn.py

    # Override specific parameters
    python scripts/train_tcn.py training.num_epochs=50 training.batch_size=16

    # Use different model variant
    python scripts/train_tcn.py model=tcn_small

    # Hyperparameter sweep
    python scripts/train_tcn.py -m model.architecture.num_channels=[15,15,15],[25,25,25,25,25]
"""

from exoskeleton_ml.utils import (
    EarlyStopping,
    compute_metrics,
    get_device,
    save_best_model,
    save_checkpoint,
)
from exoskeleton_ml.models import create_model
from exoskeleton_ml.data import create_dataloaders
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def masked_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute loss only on non-padded and non-NaN positions.

    Args:
        outputs: Model predictions (batch, seq_len, 4).
        targets: Ground truth (batch, seq_len, 4).
        mask: Validity mask (batch, seq_len).
        criterion: Loss function.

    Returns:
        Masked loss value.
    """
    # Expand mask to match output dimensions: (batch, seq_len, 4)
    mask_expanded = mask.unsqueeze(-1).expand_as(targets)

    # Create combined mask: valid positions AND non-NaN values in both outputs and targets
    nan_mask_targets = ~torch.isnan(targets)
    nan_mask_outputs = ~torch.isnan(outputs)
    valid_mask = mask_expanded & nan_mask_targets & nan_mask_outputs

    # Convert boolean mask to float
    valid_mask_float = valid_mask.float()

    # Compute squared error
    squared_error = (outputs - targets) ** 2

    # Apply mask to the squared error
    masked_squared_error = squared_error * valid_mask_float

    # Count valid elements
    num_valid = valid_mask_float.sum()

    # Check if we have any valid data
    if num_valid == 0:
        # Return zero loss if no valid data
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    # Compute mean squared error over valid positions only
    loss = masked_squared_error.sum() / num_valid

    return loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch.

    Args:
        model: PyTorch model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        epoch: Current epoch number.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch["inputs"].to(device)  # (batch, seq_len, 28)
        targets = batch["targets"].to(device)  # (batch, seq_len, 4)
        mask = batch["mask"].to(device)  # (batch, seq_len)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len, 4)

        # Compute masked loss
        loss = masked_loss(outputs, targets, mask, criterion)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict]:
    """Validate on validation set.

    Args:
        model: PyTorch model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        epoch: Current epoch number.

    Returns:
        Tuple of (average loss, metrics dictionary).
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    all_masks = []

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)

    for batch in progress_bar:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = masked_loss(outputs, targets, mask, criterion)
        total_loss += loss.item()

        # Store for metrics computation - flatten valid positions only
        # This avoids issues with different sequence lengths across batches
        for i in range(outputs.shape[0]):  # Iterate over batch
            seq_len = mask[i].sum().item()  # Get actual length for this sequence
            all_outputs.append(outputs[i, :seq_len].cpu())  # Only valid timesteps
            all_targets.append(targets[i, :seq_len].cpu())
            all_masks.append(mask[i, :seq_len].cpu())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute average loss
    avg_loss = total_loss / len(loader)

    # Concatenate all sequences (now all are 1D after flattening)
    all_outputs = torch.cat(all_outputs, dim=0)  # (total_valid_timesteps, 4)
    all_targets = torch.cat(all_targets, dim=0)  # (total_valid_timesteps, 4)
    all_masks = torch.cat(all_masks, dim=0)  # (total_valid_timesteps,)

    # Reshape for metrics computation - add batch and sequence dimensions back
    # but now all sequences are concatenated into one long sequence
    all_outputs = all_outputs.unsqueeze(0)  # (1, total_timesteps, 4)
    all_targets = all_targets.unsqueeze(0)  # (1, total_timesteps, 4)
    all_masks = all_masks.unsqueeze(0)  # (1, total_timesteps)

    # Compute metrics
    metrics = compute_metrics(all_outputs, all_targets, all_masks)

    return avg_loss, metrics


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("TCN Training for Exoskeleton Joint Moment Estimation")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Setup
    device = get_device()
    print(f"\nDevice: {device}")

    output_dir = Path(cfg.training.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Create dataloaders
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        hf_repo=cfg.data.hf_repo,
        cache_dir=cfg.data.cache_dir,
        train_participants=cfg.data.splits.train,
        val_participants=cfg.data.splits.val,
        test_participants=cfg.data.splits.test,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get("num_workers", 4),
        normalize=cfg.data.preprocessing.get("normalize", True),
        device=device.type,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    model = create_model(cfg.model).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model.name}")
    print(f"Parameters: {num_params:,}")

    if hasattr(model, "get_effective_history"):
        eff_hist = model.get_effective_history()
        print(
            f"Effective history: {eff_hist} timesteps ({eff_hist/100:.2f}s at 100Hz)")

    # Loss function
    criterion = nn.MSELoss(reduction="mean")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.0001),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.get("early_stopping_patience", 20),
        min_delta=0.0,
        mode="min",
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch)
        val_losses.append(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val RMSE:   {val_metrics['rmse_overall']:.4f} Nm/kg")
        print(f"Val R²:     {val_metrics['r2_overall']:.4f}")
        print(f"Val MAE:    {val_metrics['mae_overall']:.4f} Nm/kg")

        # Per-joint metrics
        print(f"  Hip L:  {val_metrics['rmse_hip_l']:.4f} Nm/kg")
        print(f"  Hip R:  {val_metrics['rmse_hip_r']:.4f} Nm/kg")
        print(f"  Knee L: {val_metrics['rmse_knee_l']:.4f} Nm/kg")
        print(f"  Knee R: {val_metrics['rmse_knee_r']:.4f} Nm/kg")

        # Learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.2e}")

        # Scheduler step
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(
                model,
                optimizer,
                epoch,
                val_loss,
                output_dir,
                metrics=val_metrics,
            )
            print(f"✅ New best model saved! (Val Loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % cfg.training.get("save_frequency", 10) == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                output_dir / f"checkpoint_epoch_{epoch}.pt",
                scheduler=scheduler,
                metrics=val_metrics,
            )

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    # Load best model
    from exoskeleton_ml.utils import load_checkpoint

    load_checkpoint(
        output_dir / "best_model.pt",
        model,
        device=str(device),
    )

    # Evaluate
    test_loss, test_metrics = validate(
        model, test_loader, criterion, device, 0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_metrics['rmse_overall']:.4f} Nm/kg")
    print(f"Test R²:   {test_metrics['r2_overall']:.4f}")
    print(f"Test MAE:  {test_metrics['mae_overall']:.4f} Nm/kg")
    print("\nPer-joint Test RMSE:")
    print(f"  Hip L:  {test_metrics['rmse_hip_l']:.4f} Nm/kg")
    print(f"  Hip R:  {test_metrics['rmse_hip_r']:.4f} Nm/kg")
    print(f"  Knee L: {test_metrics['rmse_knee_l']:.4f} Nm/kg")
    print(f"  Knee R: {test_metrics['rmse_knee_r']:.4f} Nm/kg")

    # Save final results
    results = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    torch.save(results, output_dir / "training_results.pt")
    print(f"\n✅ Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    train()
