"""Evaluation metrics for regression tasks."""

from typing import Dict

import torch


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
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
) -> Dict[str, Dict[str, float]]:
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
