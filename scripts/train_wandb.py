"""
Training script for models on exoskeleton joint moment estimation.

Usage:
    # Basic training with default config
    python scripts/train.py

    # Override specific parameters
    python scripts/train.py training.num_epochs=50 training.batch_size=16

    # Use different model variant
    python scripts/train.py model=tcn_small

    # Hyperparameter sweep
    python scripts/train.py -m model.architecture.num_channels=[15,15,15],[25,25,25,25,25]
"""

from exoskeleton_ml.utils import (
    EarlyStopping,
    RunningMetrics,
    get_device,
    save_best_model,
    save_checkpoint,
)
from exoskeleton_ml.models import create_model
from exoskeleton_ml.data import create_dataloaders
from exoskeleton_ml.training import Trainer
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=os.getenv("WANDB_ENTITY"),
        # Set the wandb project where this run will be logged.
        project=os.getenv("WANDB_PROJECT"),

        # Track hyperparameters and run metadata.
        config=dict(cfg),
    )

    run.define_metric("epoch", hidden=True)
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")

    print("=" * 80)
    print(f"{cfg.model.name} Training for Exoskeleton Joint Moment Estimation")
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

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
        print("-" * 80)

        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_metrics = trainer.validate(val_loader, epoch)
        val_losses.append(val_loss)

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/rmse": val_metrics["rmse_overall"],
            "val/mae": val_metrics["mae_overall"],
            "val/r2": val_metrics["r2_overall"],
            "val/nrmse": val_metrics["nrmse_overall"],
            "val/rmse_hip_l": val_metrics["rmse_hip_l"],
            "val/rmse_hip_r": val_metrics["rmse_hip_r"],
            "val/rmse_knee_l": val_metrics["rmse_knee_l"],
            "val/rmse_knee_r": val_metrics["rmse_knee_r"],
            "epoch": epoch,
        })
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val RMSE:   {val_metrics['rmse_overall']:.4f} Nm/kg")
        print(f"Val Ru00b2:     {val_metrics['r2_overall']:.4f}")
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
            print(f"u2705 New best model saved! (Val Loss: {val_loss:.4f})")

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
            print(f"\nu26a0ufe0f  Early stopping triggered at epoch {epoch}")
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
    test_loss, test_metrics = trainer.validate(test_loader, 0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_metrics['rmse_overall']:.4f} Nm/kg")
    print(f"Test Ru00b2:   {test_metrics['r2_overall']:.4f}")
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
    print(f"\nu2705 Training complete! Results saved to {output_dir}")

    run.summary["test/loss"] = test_loss
    run.summary["test/rmse"] = test_metrics["rmse_overall"]
    run.summary["test/mae"] = test_metrics["mae_overall"]
    run.summary["test/r2"] = test_metrics["r2_overall"]
    run.summary["test/nrmse"] = test_metrics["nrmse_overall"]
    run.summary["test/rmse_hip_l"] = test_metrics["rmse_hip_l"]
    run.summary["test/rmse_hip_r"] = test_metrics["rmse_hip_r"]
    run.summary["test/rmse_knee_l"] = test_metrics["rmse_knee_l"]
    run.summary["test/rmse_knee_r"] = test_metrics["rmse_knee_r"]
    run.finish()


if __name__ == "__main__":
    train()
