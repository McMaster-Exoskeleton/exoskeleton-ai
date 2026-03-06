"""Quick diagnostic script to check for data issues."""

import torch
from datasets import load_from_disk

# Load HF dataset
hf_cache = "data/processed/phase1/hf_cache"
dataset = load_from_disk(hf_cache)

print(f"Total trials: {len(dataset)}")

# Check first few trials for issues
for i in range(min(5, len(dataset))):
    trial = dataset[i]

    print(f"\nTrial {i}: {trial['participant']} - {trial['trial_name']}")
    print(f"  Sequence length: {trial['sequence_length']}")

    # Convert to tensors
    imu_features = torch.tensor(trial["imu_features"], dtype=torch.float32)
    angle_features = torch.tensor(trial["angle_features"], dtype=torch.float32)
    moment_targets = torch.tensor(trial["moment_targets"], dtype=torch.float32)

    # Check for NaN/inf
    print(f"  IMU features - has NaN: {torch.isnan(imu_features).any().item()}, has inf: {torch.isinf(imu_features).any().item()}")
    print(f"  Angle features - has NaN: {torch.isnan(angle_features).any().item()}, has inf: {torch.isinf(angle_features).any().item()}")
    print(f"  Moment targets - has NaN: {torch.isnan(moment_targets).any().item()}, has inf: {torch.isinf(moment_targets).any().item()}")

    # Check ranges
    print(f"  IMU features range: [{imu_features.min().item():.2f}, {imu_features.max().item():.2f}]")
    print(f"  Angle features range: [{angle_features.min().item():.2f}, {angle_features.max().item():.2f}]")
    print(f"  Moment targets range: [{moment_targets.min().item():.2f}, {moment_targets.max().item():.2f}]")

    # Check feature-wise stats
    print(f"  IMU feature 0 - mean: {imu_features[:, 0].mean().item():.4f}, std: {imu_features[:, 0].std().item():.4f}")
    print(f"  Angle feature 0 - mean: {angle_features[:, 0].mean().item():.4f}, std: {angle_features[:, 0].std().item():.4f}")

print("\n" + "="*60)
print("Checking across all trials...")

# Sample 10 random trials
import random
sample_indices = random.sample(range(len(dataset)), min(10, len(dataset)))

all_has_nan = False
all_has_inf = False

for i in sample_indices:
    trial = dataset[i]
    imu_features = torch.tensor(trial["imu_features"], dtype=torch.float32)
    angle_features = torch.tensor(trial["angle_features"], dtype=torch.float32)
    moment_targets = torch.tensor(trial["moment_targets"], dtype=torch.float32)

    if torch.isnan(imu_features).any() or torch.isnan(angle_features).any() or torch.isnan(moment_targets).any():
        all_has_nan = True
        print(f"Trial {i} has NaN values")
    if torch.isinf(imu_features).any() or torch.isinf(angle_features).any() or torch.isinf(moment_targets).any():
        all_has_inf = True
        print(f"Trial {i} has inf values")

if not all_has_nan and not all_has_inf:
    print("✅ No NaN or inf values found in sampled trials")
else:
    print(f"⚠️  Found issues: NaN={all_has_nan}, inf={all_has_inf}")
