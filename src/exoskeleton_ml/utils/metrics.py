"""Evaluation metrics for regression tasks."""


import torch


class RunningMetrics:
    """Online/streaming computation of regression metrics without storing all data.

    This class accumulates statistics incrementally to avoid memory issues
    when computing metrics over large datasets across many epochs.
    """

    def __init__(self, num_joints: int = 4):
        """Initialize running metrics tracker.

        Args:
            num_joints: Number of output joints (default 4).
        """
        self.num_joints = num_joints
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        # Overall statistics
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_targets = 0.0
        self.sum_targets_squared = 0.0
        self.count = 0
        self.target_min = float('inf')
        self.target_max = float('-inf')

        # Per-joint statistics
        self.joint_sse = [0.0] * self.num_joints
        self.joint_count = [0] * self.num_joints

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        """Update running statistics with a new batch.

        Args:
            predictions: Model predictions of shape (batch, seq_len, num_joints).
            targets: Ground truth of shape (batch, seq_len, num_joints).
            mask: Validity mask of shape (batch, seq_len).
        """
        # Expand mask to match predictions shape
        mask_expanded = mask.unsqueeze(-1).expand_as(predictions)

        # Combined validity mask
        valid_mask = mask_expanded & ~torch.isnan(predictions) & ~torch.isnan(targets)

        # Extract valid values
        pred_valid = predictions[valid_mask]
        target_valid = targets[valid_mask]

        if pred_valid.numel() == 0:
            return

        # Update overall statistics
        errors = pred_valid - target_valid
        self.sum_squared_error += (errors ** 2).sum().item()
        self.sum_absolute_error += errors.abs().sum().item()
        self.sum_targets += target_valid.sum().item()
        self.sum_targets_squared += (target_valid ** 2).sum().item()
        self.count += pred_valid.numel()
        self.target_min = min(self.target_min, target_valid.min().item())
        self.target_max = max(self.target_max, target_valid.max().item())

        # Update per-joint statistics
        for j in range(self.num_joints):
            joint_mask = valid_mask[..., j]
            if joint_mask.sum() > 0:
                pred_j = predictions[..., j][joint_mask]
                target_j = targets[..., j][joint_mask]
                self.joint_sse[j] += ((pred_j - target_j) ** 2).sum().item()
                self.joint_count[j] += pred_j.numel()

    def compute(self) -> dict[str, float]:
        """Compute final metrics from accumulated statistics.

        Returns:
            Dictionary of computed metrics.
        """
        if self.count == 0:
            return {
                "rmse_overall": 0.0,
                "rmse_hip_l": 0.0,
                "rmse_hip_r": 0.0,
                "rmse_knee_l": 0.0,
                "rmse_knee_r": 0.0,
                "mae_overall": 0.0,
                "r2_overall": 0.0,
                "nrmse_overall": 0.0,
            }

        # Overall metrics
        mse = self.sum_squared_error / self.count
        rmse = mse ** 0.5
        mae = self.sum_absolute_error / self.count

        # R² computation using Welford's online variance formula
        mean_target = self.sum_targets / self.count
        # Var = E[X²] - E[X]²
        var_target = (self.sum_targets_squared / self.count) - (mean_target ** 2)
        ss_tot = var_target * self.count
        ss_res = self.sum_squared_error
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # NRMSE
        target_range = self.target_max - self.target_min
        nrmse = rmse / target_range if target_range > 0 else 0.0

        # Per-joint RMSE
        joint_names = ["hip_l", "hip_r", "knee_l", "knee_r"]
        per_joint_rmse = {}
        for j, name in enumerate(joint_names):
            if self.joint_count[j] > 0:
                joint_mse = self.joint_sse[j] / self.joint_count[j]
                per_joint_rmse[f"rmse_{name}"] = joint_mse ** 0.5
            else:
                per_joint_rmse[f"rmse_{name}"] = 0.0

        return {
            "rmse_overall": rmse,
            **per_joint_rmse,
            "mae_overall": mae,
            "r2_overall": r2,
            "nrmse_overall": nrmse,
        }


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute regression metrics for joint moment estimation.

    Args:
        predictions: Model predictions of shape (batch, seq_len, 4).
        targets: Ground truth targets of shape (batch, seq_len, 4).
        mask: Boolean mask of shape (batch, seq_len) indicating valid positions.

    Returns:
        Dictionary containing:
            - rmse_overall: Overall RMSE across all joints
            - rmse_hip_l, rmse_hip_r, rmse_knee_l, rmse_knee_r: Per-joint RMSE
            - mae_overall: Overall Mean Absolute Error
            - r2_overall: Overall R² score
            - nrmse_overall: Normalized RMSE (by target range)
    """
    # Expand mask to match predictions shape: (batch, seq_len, 1)
    mask = mask.unsqueeze(-1)

    # Create combined mask: valid positions AND non-NaN values
    valid_mask = mask & ~torch.isnan(predictions) & ~torch.isnan(targets)

    # Extract only valid values
    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]

    # Prevent division by zero
    num_valid = valid_mask.sum()
    if num_valid == 0:
        return {
            "rmse_overall": 0.0,
            "rmse_hip_l": 0.0,
            "rmse_hip_r": 0.0,
            "rmse_knee_l": 0.0,
            "rmse_knee_r": 0.0,
            "mae_overall": 0.0,
            "r2_overall": 0.0,
            "nrmse_overall": 0.0,
        }

    # Overall RMSE
    mse = ((pred_valid - target_valid) ** 2).mean()
    rmse = torch.sqrt(mse).item()

    # Per-joint RMSE
    joint_names = ["hip_l", "hip_r", "knee_l", "knee_r"]
    per_joint_rmse = {}
    for i, joint in enumerate(joint_names):
        # Get valid mask for this specific joint
        joint_mask = valid_mask[..., i]
        if joint_mask.sum() > 0:
            pred_joint = predictions[..., i][joint_mask]
            target_joint = targets[..., i][joint_mask]
            joint_mse = ((pred_joint - target_joint) ** 2).mean()
            per_joint_rmse[f"rmse_{joint}"] = torch.sqrt(joint_mse).item()
        else:
            per_joint_rmse[f"rmse_{joint}"] = 0.0

    # MAE (Mean Absolute Error)
    mae = (pred_valid - target_valid).abs().mean()

    # R² Score (Coefficient of Determination)
    ss_res = ((target_valid - pred_valid) ** 2).sum()
    ss_tot = ((target_valid - target_valid.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)

    # NRMSE (Normalized RMSE by range)
    target_range = target_valid.max() - target_valid.min()
    nrmse = rmse / target_range.item() if target_range.item() > 0 else 0.0

    return {
        "rmse_overall": rmse,
        **per_joint_rmse,
        "mae_overall": mae.item(),
        "r2_overall": r2.item(),
        "nrmse_overall": nrmse,
    }


def compute_per_participant_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    participants: list,
) -> dict[str, dict[str, float]]:
    """Compute metrics separately for each participant.

    Args:
        predictions: Model predictions of shape (batch, seq_len, 4).
        targets: Ground truth targets of shape (batch, seq_len, 4).
        mask: Boolean mask of shape (batch, seq_len).
        participants: List of participant IDs for each sample in the batch.

    Returns:
        Dictionary mapping participant ID to their metrics dictionary.
    """
    participant_metrics = {}
    unique_participants = set(participants)

    for participant in unique_participants:
        # Get indices for this participant
        indices = [i for i, p in enumerate(participants) if p == participant]

        if not indices:
            continue

        # Extract data for this participant
        participant_preds = predictions[indices]
        participant_targets = targets[indices]
        participant_mask = mask[indices]

        # Compute metrics
        metrics = compute_metrics(participant_preds, participant_targets, participant_mask)
        participant_metrics[participant] = metrics

    return participant_metrics
