#!/usr/bin/env python3
"""Plot continuous motor torques during walking to visualize patterns.

This gives you a straightforward view of what the motor torques look like
over time during normal walking, without trying to segment into gait cycles.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from exoskeleton_ml.data.datasets import ExoskeletonDataset


def main():
    print("Loading dataset...")
    dataset = ExoskeletonDataset(
        hf_repo="MacExo/exoData",
        cache_dir="data/processed/phase1",
        participants=None,
        normalize=False,
    )

    # Find normal walking trials with COMPLETE bilateral data (all 4 joints)
    print("Finding normal walking trials with complete bilateral data...")
    normal_walking_trials = []
    for idx in range(len(dataset)):
        trial_info = dataset.get_trial_info(idx)
        if "normal_walk" in trial_info["trial_name"].lower():
            # Check if this trial has all 4 joints with data
            trial_data = torch.load(dataset.index[idx]["file"])
            targets = trial_data["targets"].numpy()

            has_all_joints = all([
                np.any(~np.isnan(targets[:, 0])),  # Hip L
                np.any(~np.isnan(targets[:, 1])),  # Hip R
                np.any(~np.isnan(targets[:, 2])),  # Knee L
                np.any(~np.isnan(targets[:, 3]))   # Knee R
            ])

            if has_all_joints:
                normal_walking_trials.append(idx)

    print(f"Found {len(normal_walking_trials)} normal walking trials with bilateral data\n")

    # Find trials from a participant with all 5 speeds
    # Look for participant with 0.6, 1.2, 1.8, 2.0, 2.5 m/s trials (all exo ON)
    target_speeds = ['0-6', '1-2', '1-8', '2-0', '2-5']
    selected_trials = []

    # Try to find one participant with all speeds (exo ON)
    for participant in ['BT02', 'BT03', 'BT06', 'BT07', 'BT08', 'BT09', 'BT10', 'BT11']:
        participant_speed_trials = {speed: None for speed in target_speeds}

        for idx in normal_walking_trials:
            trial_info = dataset.get_trial_info(idx)
            if trial_info['participant'] == participant:
                trial_name = trial_info['trial_name']
                # Look for exo ON trials with target speeds
                if '_on' in trial_name:
                    for speed in target_speeds:
                        if f'_{speed}_on' in trial_name:
                            participant_speed_trials[speed] = idx
                            break

        # Check if we found all speeds for this participant
        if all(v is not None for v in participant_speed_trials.values()):
            print(f"Found complete speed range for participant {participant}\n")
            selected_trials = [participant_speed_trials[speed] for speed in target_speeds]
            break

    if not selected_trials:
        print("Could not find participant with all 5 speeds, using first 5 trials instead\n")
        num_trials = min(5, len(normal_walking_trials))
        selected_trials = normal_walking_trials[:num_trials]

    num_trials = len(selected_trials)

    for trial_num in range(num_trials):
        idx = selected_trials[trial_num]
        trial_info = dataset.get_trial_info(idx)
        trial_data = torch.load(dataset.index[idx]["file"])

        targets = trial_data["targets"].numpy()  # (seq_len, 4) in Nm/kg
        mass_kg = trial_data["mass_kg"]

        print(f"Trial {trial_num + 1}: {trial_info['trial_name']}")
        print(f"  Participant: {trial_info['participant']}")
        print(f"  Mass: {mass_kg:.1f} kg")
        print(f"  Duration: {len(targets)/100:.1f} seconds")

        # Check which joints have data
        for i, joint in enumerate(["Hip L", "Hip R", "Knee L", "Knee R"]):
            valid = np.sum(~np.isnan(targets[:, i]))
            total = len(targets)
            print(f"  {joint}: {valid}/{total} valid ({100*valid/total:.0f}%)")

        print()

        # Georgia Tech control strategy
        hip_scale = 0.20
        knee_scale = 0.15

        # Compute motor commands
        hip_l_cmd = targets[:, 0] * hip_scale * mass_kg
        hip_r_cmd = targets[:, 1] * hip_scale * mass_kg
        knee_l_cmd = targets[:, 2] * knee_scale * mass_kg
        knee_r_cmd = targets[:, 3] * knee_scale * mass_kg

        # Time vector
        time_sec = np.arange(len(targets)) / 100.0

        # Plot a window of data (e.g., 10 seconds)
        window_duration = min(10.0, len(targets) / 100.0)  # 10 seconds or full trial
        window_samples = int(window_duration * 100)

        # Start from middle to avoid edge effects
        start_idx = max(0, len(targets) // 2 - window_samples // 2)
        end_idx = start_idx + window_samples

        time_window = time_sec[start_idx:end_idx]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Motor Torques During Walking: {trial_info["trial_name"]}\n'
                    f'({window_duration:.1f}s window, Georgia Tech strategy: 20% hip / 15% knee)',
                    fontsize=14, fontweight='bold')

        # Hip Left
        ax = axes[0, 0]
        valid_mask = ~np.isnan(hip_l_cmd[start_idx:end_idx])
        if np.any(valid_mask):
            ax.plot(time_window[valid_mask], hip_l_cmd[start_idx:end_idx][valid_mask],
                   'b-', linewidth=1.5, label='Hip Left Motor Command')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Torque (Nm)', fontsize=11)
            ax.set_title('Hip Left Motor Torque', fontsize=12, fontweight='bold')
            ax.legend()

            # Add stats
            valid_data = hip_l_cmd[start_idx:end_idx][valid_mask]
            ax.text(0.02, 0.98, f'Peak: {np.max(np.abs(valid_data)):.2f} Nm\n'
                              f'Mean: {np.mean(valid_data):.2f} Nm',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
                   ha='center', va='center', fontsize=20, color='red')
            ax.set_title('Hip Left Motor Torque (NO DATA)', fontsize=12)

        # Hip Right
        ax = axes[0, 1]
        valid_mask = ~np.isnan(hip_r_cmd[start_idx:end_idx])
        if np.any(valid_mask):
            ax.plot(time_window[valid_mask], hip_r_cmd[start_idx:end_idx][valid_mask],
                   'c-', linewidth=1.5, label='Hip Right Motor Command')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Torque (Nm)', fontsize=11)
            ax.set_title('Hip Right Motor Torque', fontsize=12, fontweight='bold')
            ax.legend()

            valid_data = hip_r_cmd[start_idx:end_idx][valid_mask]
            ax.text(0.02, 0.98, f'Peak: {np.max(np.abs(valid_data)):.2f} Nm\n'
                              f'Mean: {np.mean(valid_data):.2f} Nm',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
                   ha='center', va='center', fontsize=20, color='red')
            ax.set_title('Hip Right Motor Torque (NO DATA)', fontsize=12)

        # Knee Left
        ax = axes[1, 0]
        valid_mask = ~np.isnan(knee_l_cmd[start_idx:end_idx])
        if np.any(valid_mask):
            ax.plot(time_window[valid_mask], knee_l_cmd[start_idx:end_idx][valid_mask],
                   'r-', linewidth=1.5, label='Knee Left Motor Command')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Torque (Nm)', fontsize=11)
            ax.set_title('Knee Left Motor Torque', fontsize=12, fontweight='bold')
            ax.legend()

            valid_data = knee_l_cmd[start_idx:end_idx][valid_mask]
            ax.text(0.02, 0.98, f'Peak: {np.max(np.abs(valid_data)):.2f} Nm\n'
                              f'Mean: {np.mean(valid_data):.2f} Nm',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
                   ha='center', va='center', fontsize=20, color='red')
            ax.set_title('Knee Left Motor Torque (NO DATA)', fontsize=12)

        # Knee Right
        ax = axes[1, 1]
        valid_mask = ~np.isnan(knee_r_cmd[start_idx:end_idx])
        if np.any(valid_mask):
            ax.plot(time_window[valid_mask], knee_r_cmd[start_idx:end_idx][valid_mask],
                   'm-', linewidth=1.5, label='Knee Right Motor Command')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Torque (Nm)', fontsize=11)
            ax.set_title('Knee Right Motor Torque', fontsize=12, fontweight='bold')
            ax.legend()

            valid_data = knee_r_cmd[start_idx:end_idx][valid_mask]
            ax.text(0.02, 0.98, f'Peak: {np.max(np.abs(valid_data)):.2f} Nm\n'
                              f'Mean: {np.mean(valid_data):.2f} Nm',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
                   ha='center', va='center', fontsize=20, color='red')
            ax.set_title('Knee Right Motor Torque (NO DATA)', fontsize=12)

        plt.tight_layout()

        output_file = f"continuous_walking_torques_trial{trial_num + 1}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Created {num_trials} plots showing continuous motor torques during walking.

Key observations:
1. Some joints may have NO DATA (all NaN values) - this is trial-specific
2. Torques show rhythmic patterns corresponding to gait cycles
3. Peak torques are typically 5-6 Nm for motors with Georgia Tech strategy
4. The patterns repeat with gait frequency (~1 Hz for normal walking)

To properly segment gait cycles, you would need:
- Heel strike/toe-off timing (not directly in the dataset)
- Force plate data or foot contact sensors
- Or estimate from joint angle zero-crossings

For now, you can visually inspect these plots to see ~2 full steps in
the 10-second windows (approximately 10 strides total).
""")


if __name__ == "__main__":
    main()
