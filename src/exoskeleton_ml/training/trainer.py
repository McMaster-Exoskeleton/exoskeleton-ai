"""Training loop implementation."""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from exoskeleton_ml.utils import RunningMetrics

from exoskeleton_ml.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Dummy Trainer so Far"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir

    def train_epoch(
        self,
        loader: DataLoader,
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
        self.model.train()
        total_loss = 0.0
        num_batches = len(loader)

        progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch["inputs"].to(self.device)  # (batch, seq_len, 28)
            targets = batch["targets"].to(self.device)  # (batch, seq_len, 4)
            mask = batch["mask"].to(self.device)  # (batch, seq_len)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # (batch, seq_len, 4)

            # Compute masked loss
            loss = self.masked_loss(outputs, targets, mask)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
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
        self.model.eval()
        total_loss = 0.0

        # Use running metrics to avoid memory accumulation
        running_metrics = RunningMetrics(num_joints=4)

        progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)

        for batch in progress_bar:
            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = Trainer.masked_loss(outputs, targets, mask)
            total_loss += loss.item()

            # Update running metrics (no tensor accumulation)
            running_metrics.update(outputs, targets, mask)

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute average loss
        avg_loss = total_loss / len(loader)

        # Compute final metrics from accumulated statistics
        metrics = running_metrics.compute()

        return avg_loss, metrics

    @staticmethod
    def masked_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss only on non-padded and non-NaN positions.

        Args:
            outputs: Model predictions (batch, seq_len, 4).
            targets: Ground truth (batch, seq_len, 4).
            mask: Validity mask (batch, seq_len).

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
