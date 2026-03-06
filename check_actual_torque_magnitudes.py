#!/usr/bin/env python3
"""Check the actual magnitude of biological moments and motor commands."""

import numpy as np
import torch
from exoskeleton_ml.data.datasets import ExoskeletonDataset


def main():
    print("Loading dataset...")
    dataset = ExoskeletonDataset(
        hf_repo="MacExo/exoData",
        cache_dir="data/processed/phase1",
        participants=None,
        normalize=False,
    )

    # Find one normal walking trial
    for idx in range(len(dataset)):
        trial_info = dataset.get_trial_info(idx)
        if "normal_walk" in trial_info["trial_name"].lower():
            break

    print(f"\nAnalyzing trial: {trial_info['trial_name']}")
    trial_data = torch.load(dataset.index[idx]["file"])

    targets = trial_data["targets"].numpy()  # (seq_len, 4) in Nm/kg
    mass_kg = trial_data["mass_kg"]

    print(f"Participant mass: {mass_kg:.1f} kg")
    print(f"Trial length: {len(targets)} samples ({len(targets)/100:.1f} seconds at 100 Hz)")
    print()

    # Show raw biological moments (Nm/kg)
    print("=" * 80)
    print("BIOLOGICAL MOMENTS (Nm/kg - as stored in dataset)")
    print("=" * 80)

    joint_names = ["Hip Left", "Hip Right", "Knee Left", "Knee Right"]
    for i, name in enumerate(joint_names):
        values = targets[:, i]
        valid = values[~np.isnan(values)]

        print(f"\n{name}:")
        print(f"  Valid samples: {len(valid)}/{len(values)} ({100*len(valid)/len(values):.1f}%)")
        if len(valid) > 0:
            print(f"  Mean:     {np.mean(valid):>8.3f} Nm/kg")
            print(f"  Std:      {np.std(valid):>8.3f} Nm/kg")
            print(f"  Min:      {np.min(valid):>8.3f} Nm/kg")
            print(f"  Max:      {np.max(valid):>8.3f} Nm/kg")
            print(f"  |Max|:    {np.max(np.abs(valid)):>8.3f} Nm/kg")
        else:
            print("  ALL VALUES ARE NaN!")

    # Convert to actual Nm
    print()
    print("=" * 80)
    print("BIOLOGICAL MOMENTS (Nm - actual torques)")
    print("=" * 80)

    for i, name in enumerate(joint_names):
        values_nmkg = targets[:, i]
        values_nm = values_nmkg * mass_kg
        valid = values_nm[~np.isnan(values_nm)]

        print(f"\n{name}:")
        if len(valid) > 0:
            print(f"  Mean:     {np.mean(valid):>8.3f} Nm")
            print(f"  Std:      {np.std(valid):>8.3f} Nm")
            print(f"  Min:      {np.min(valid):>8.3f} Nm")
            print(f"  Max:      {np.max(valid):>8.3f} Nm")
            print(f"  |Max|:    {np.max(np.abs(valid)):>8.3f} Nm")
        else:
            print("  ALL VALUES ARE NaN!")

    # Georgia Tech control strategy
    hip_scale = 0.20
    knee_scale = 0.15

    print()
    print("=" * 80)
    print("MOTOR COMMANDS with Georgia Tech Strategy (20% hip, 15% knee)")
    print("=" * 80)

    hip_l_cmd = targets[:, 0] * hip_scale * mass_kg
    hip_r_cmd = targets[:, 1] * hip_scale * mass_kg
    knee_l_cmd = targets[:, 2] * knee_scale * mass_kg
    knee_r_cmd = targets[:, 3] * knee_scale * mass_kg

    commands = [
        ("Hip Left", hip_l_cmd),
        ("Hip Right", hip_r_cmd),
        ("Knee Left", knee_l_cmd),
        ("Knee Right", knee_r_cmd)
    ]

    for name, cmd in commands:
        valid = cmd[~np.isnan(cmd)]
        print(f"\n{name}:")
        if len(valid) > 0:
            print(f"  Mean:     {np.mean(valid):>8.3f} Nm")
            print(f"  Std:      {np.std(valid):>8.3f} Nm")
            print(f"  Min:      {np.min(valid):>8.3f} Nm")
            print(f"  Max:      {np.max(valid):>8.3f} Nm")
            print(f"  |Max|:    {np.max(np.abs(valid)):>8.3f} Nm")
        else:
            print("  ALL VALUES ARE NaN!")

    # Show a small sample of the data
    print()
    print("=" * 80)
    print("SAMPLE DATA (first 20 timesteps)")
    print("=" * 80)
    print()
    print("Time(s)  | Hip_L(Nm/kg) | Hip_R(Nm/kg) | Knee_L(Nm/kg) | Knee_R(Nm/kg) | Hip_L_cmd(Nm) | Knee_L_cmd(Nm)")
    print("-" * 110)

    for i in range(min(20, len(targets))):
        t = i / 100.0
        print(f"{t:>6.2f}  | {targets[i,0]:>12.3f} | {targets[i,1]:>12.3f} | "
              f"{targets[i,2]:>13.3f} | {targets[i,3]:>13.3f} | "
              f"{hip_l_cmd[i]:>13.3f} | {knee_l_cmd[i]:>14.3f}")


if __name__ == "__main__":
    main()
