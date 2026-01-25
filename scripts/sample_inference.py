"""
Simple inference script for testing model deployment on Raspberry Pi.

Usage:
    python scripts/raspberry_pi_inference.py
    python scripts/raspberry_pi_inference.py --model-path outputs/path/to/best_model.pt
    python scripts/raspberry_pi_inference.py --benchmark --num-runs 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from exoskeleton_ml.models import create_model
from exoskeleton_ml.utils import load_checkpoint
from omegaconf import OmegaConf


def create_sample_input(
    sequence_length: int = 100,
    num_features: int = 28,
    batch_size: int = 1,
) -> torch.Tensor:
    return torch.randn(batch_size, sequence_length, num_features)


def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location=device)

    config_path = model_path.parent / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file not found at {config_path}")

    config = OmegaConf.load(config_path)
    model_config = config.model

    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def run_inference(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, float]:
    input_data = input_data.to(device)

    with torch.no_grad():
        start_time = time.perf_counter()
        predictions = model(input_data)
        end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000
    return predictions, inference_time_ms


def benchmark_inference(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    num_runs: int = 100,
    device: str = "cpu",
) -> dict:
    print(f"\nRunning benchmark with {num_runs} iterations...")

    inference_times = []

    for i in range(num_runs):
        _, inference_time = run_inference(model, input_data, device)
        inference_times.append(inference_time)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_runs}")

    inference_times = np.array(inference_times)

    stats = {
        "mean_ms": np.mean(inference_times),
        "std_ms": np.std(inference_times),
        "min_ms": np.min(inference_times),
        "max_ms": np.max(inference_times),
        "median_ms": np.median(inference_times),
        "p95_ms": np.percentile(inference_times, 95),
        "p99_ms": np.percentile(inference_times, 99),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Test model inference for Raspberry Pi deployment"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/2026-01-08/18-29-05/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Length of input sequence",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark to measure inference time statistics",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of runs for benchmarking",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Raspberry Pi Inference Test")
    print("=" * 80)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("\nAvailable models:")
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for best_model in outputs_dir.rglob("best_model.pt"):
                print(f"  {best_model}")
        return

    print(f"\nModel: {model_path}")
    print(f"Device: {args.device}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Batch Size: {args.batch_size}")

    print("\nLoading model...")
    model = load_model(str(model_path), device=args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model file size: {model_size_mb:.2f} MB")

    print("\nCreating sample input data...")
    input_data = create_sample_input(
        sequence_length=args.sequence_length,
        num_features=28,
        batch_size=args.batch_size,
    )
    print(f"Input shape: {input_data.shape}")

    print("\nRunning single inference...")
    predictions, inference_time = run_inference(model, input_data, args.device)
    print(f"Output shape: {predictions.shape}")
    print(f"Inference time: {inference_time:.2f} ms")

    samples_per_second = 1000 / inference_time
    print(f"Throughput: {samples_per_second:.2f} sequences/second")

    if args.benchmark:
        stats = benchmark_inference(
            model, input_data, num_runs=args.num_runs, device=args.device
        )

        print("\n" + "=" * 80)
        print("Benchmark Results")
        print("=" * 80)
        print(f"Mean:   {stats['mean_ms']:.2f} ms")
        print(f"Median: {stats['median_ms']:.2f} ms")
        print(f"Std:    {stats['std_ms']:.2f} ms")
        print(f"Min:    {stats['min_ms']:.2f} ms")
        print(f"Max:    {stats['max_ms']:.2f} ms")
        print(f"P95:    {stats['p95_ms']:.2f} ms")
        print(f"P99:    {stats['p99_ms']:.2f} ms")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Parameters: {num_params:,}")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Throughput: {samples_per_second:.2f} sequences/sec")

    print("\n" + "=" * 80)
    print("Example Prediction Output")
    print("=" * 80)
    print("The model predicts joint moments (torques) for 4 joints:")
    print("  - Hip Left, Hip Right, Knee Left, Knee Right")
    print(f"\nSample prediction (first 5 timesteps):")
    print(predictions[0, :5, :])
    print("\nUnits: Nm/kg (Newton-meters per kilogram of body weight)")


if __name__ == "__main__":
    main()
