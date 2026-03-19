"""
Analyze all training runs: load training_results.pt and config.yaml from outputs/
and print a summary table plus optional loss curves.

Usage:
    python scripts/analyze_training_results.py
    python scripts/analyze_training_results.py --plot   # also save loss-curve plot
"""

import argparse
from pathlib import Path

import yaml


def find_runs(outputs_dir: Path):
    """Find all run directories that contain training_results.pt and config.yaml."""
    runs = []
    for path in outputs_dir.rglob("training_results.pt"):
        parent = path.parent
        config_path = parent / "config.yaml"
        if config_path.exists():
            # Skip .hydra subdir (has config.yaml but not our merged one)
            if ".hydra" in str(parent):
                continue
            runs.append(parent)
    return sorted(runs)


def load_run(run_dir: Path):
    """Load config and training_results.pt for one run. Uses torch only when loading .pt."""
    import torch

    config_path = run_dir / "config.yaml"
    results_path = run_dir / "training_results.pt"

    config = yaml.safe_load(config_path.read_text())
    results = torch.load(results_path, weights_only=True)

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name") or model_cfg.get("type", run_dir.name)
    return {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "config": config,
        "results": results,
    }


def print_summary(runs_data: list) -> None:
    """Print a summary table of all runs."""
    if not runs_data:
        print("No runs found.")
        return

    # Header
    print()
    print("=" * 100)
    print("TRAINING RUNS SUMMARY")
    print("=" * 100)

    for d in runs_data:
        r = d["results"]
        m = r.get("test_metrics", {})
        print()
        print(f"Run:     {d['run_dir']}")
        print(f"Model:   {d['model_name']}")
        print(f"  Best val loss:  {r.get('best_val_loss', float('nan')):.6f}")
        print(f"  Test loss:      {r.get('test_loss', float('nan')):.6f}")
        print(f"  Test RMSE:      {m.get('rmse_overall', float('nan')):.4f} Nm/kg")
        print(f"  Test R²:        {m.get('r2_overall', float('nan')):.4f}")
        print(f"  Test MAE:       {m.get('mae_overall', float('nan')):.4f} Nm/kg")
        print("  Per-joint RMSE:")
        print(f"    Hip L:  {m.get('rmse_hip_l', float('nan')):.4f}  Hip R:  {m.get('rmse_hip_r', float('nan')):.4f}")
        print(f"    Knee L: {m.get('rmse_knee_l', float('nan')):.4f}  Knee R: {m.get('rmse_knee_r', float('nan')):.4f}")
        print("-" * 100)

    # Comparison table
    print()
    print("COMPARISON (test set)")
    print("-" * 100)
    print(f"{'Model':<25} {'RMSE (Nm/kg)':<18} {'R²':<12} {'MAE (Nm/kg)':<15}")
    print("-" * 100)
    for d in runs_data:
        m = d["results"].get("test_metrics", {})
        print(
            f"{d['model_name']:<25} "
            f"{m.get('rmse_overall', float('nan')):<18.4f} "
            f"{m.get('r2_overall', float('nan')):<12.4f} "
            f"{m.get('mae_overall', float('nan')):<15.4f}"
        )
    print("=" * 100)
    print()


def plot_loss_curves(runs_data: list, save_path: Path | None) -> None:
    """Plot train/val loss curves for each run."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for d in runs_data:
        r = d["results"]
        train_losses = r.get("train_losses", [])
        val_losses = r.get("val_losses", [])
        if not train_losses and not val_losses:
            continue
        epochs = range(1, len(train_losses) + 1)
        label = d["model_name"]
        if train_losses:
            ax.plot(epochs, train_losses, alpha=0.7, linestyle="--", label=f"{label} (train)")
        if val_losses:
            ax.plot(epochs, val_losses, alpha=0.9, label=f"{label} (val)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze training runs from outputs/")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing run folders (default: outputs)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot train/val loss curves and save to outputs/training_curves.png",
    )
    args = parser.parse_args()

    runs = find_runs(args.outputs_dir)
    if not runs:
        print(f"No runs found under {args.outputs_dir}")
        return 1

    runs_data = []
    for run_dir in runs:
        try:
            runs_data.append(load_run(run_dir))
        except Exception as e:
            print(f"Warning: failed to load {run_dir}: {e}")

    if runs_data:
        model_names = [d["model_name"] for d in runs_data]
        print(f"Found {len(runs_data)} run(s): {', '.join(model_names)}")
    print_summary(runs_data)

    if args.plot and runs_data:
        save_path = args.outputs_dir / "training_curves.png"
        plot_loss_curves(runs_data, save_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
