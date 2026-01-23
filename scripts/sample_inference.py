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
import onnx
import onnxruntime

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


def benchmark_pytorch(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> dict:
    """Benchmark PyTorch model inference."""
    print(f"\nBenchmarking PyTorch ({device})...")
    input_data = input_data.to(device)

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_data)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    inference_times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_data)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        inference_times.append((end - start) * 1000)

    inference_times = np.array(inference_times)
    return {
        "mean_ms": np.mean(inference_times),
        "std_ms": np.std(inference_times),
        "min_ms": np.min(inference_times),
        "max_ms": np.max(inference_times),
        "median_ms": np.median(inference_times),
        "p95_ms": np.percentile(inference_times, 95),
        "p99_ms": np.percentile(inference_times, 99),
    }


def benchmark_onnx(
    ort_session: onnxruntime.InferenceSession,
    input_data: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """Benchmark ONNX Runtime inference."""
    print(f"\nBenchmarking ONNX Runtime (CPU)...")

    input_name = ort_session.get_inputs()[0].name
    onnx_input = {input_name: input_data.numpy()}

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        _ = ort_session.run(None, onnx_input)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    inference_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = ort_session.run(None, onnx_input)
        end = time.perf_counter()
        inference_times.append((end - start) * 1000)

    inference_times = np.array(inference_times)
    return {
        "mean_ms": np.mean(inference_times),
        "std_ms": np.std(inference_times),
        "min_ms": np.min(inference_times),
        "max_ms": np.max(inference_times),
        "median_ms": np.median(inference_times),
        "p95_ms": np.percentile(inference_times, 95),
        "p99_ms": np.percentile(inference_times, 99),
    }


def benchmark_torchscript(
    scripted_model: torch.jit.ScriptModule,
    input_data: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> dict:
    """Benchmark TorchScript model inference."""
    print(f"\nBenchmarking TorchScript ({device})...")
    input_data = input_data.to(device)

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = scripted_model(input_data)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    inference_times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = scripted_model(input_data)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        inference_times.append((end - start) * 1000)

    inference_times = np.array(inference_times)
    return {
        "mean_ms": np.mean(inference_times),
        "std_ms": np.std(inference_times),
        "min_ms": np.min(inference_times),
        "max_ms": np.max(inference_times),
        "median_ms": np.median(inference_times),
        "p95_ms": np.percentile(inference_times, 95),
        "p99_ms": np.percentile(inference_times, 99),
    }


def print_comparison(pytorch_stats: dict, onnx_stats: dict, torchscript_stats: dict = None):
    """Print comparison table between PyTorch, TorchScript, and ONNX."""
    print("\n" + "=" * 80)
    print("TIMING COMPARISON: PyTorch vs TorchScript vs ONNX Runtime")
    print("=" * 80)

    if torchscript_stats:
        print(f"\n{'Metric':<10} {'PyTorch':<12} {'TorchScript':<14} {'ONNX':<12} {'TS Speedup':<12} {'ONNX Speedup':<12}")
        print("-" * 75)
        for metric, label in [("mean_ms", "Mean"), ("median_ms", "Median"),
                              ("min_ms", "Min"), ("max_ms", "Max"),
                              ("p95_ms", "P95"), ("p99_ms", "P99")]:
            pt_val = pytorch_stats[metric]
            ts_val = torchscript_stats[metric]
            onnx_val = onnx_stats[metric]
            ts_speedup = pt_val / ts_val if ts_val > 0 else 0
            onnx_speedup = pt_val / onnx_val if onnx_val > 0 else 0
            print(f"{label:<10} {pt_val:<12.3f} {ts_val:<14.3f} {onnx_val:<12.3f} {ts_speedup:.2f}x{'':<7} {onnx_speedup:.2f}x")
    else:
        print(f"\n{'Metric':<12} {'PyTorch (ms)':<15} {'ONNX (ms)':<15} {'Speedup':<10}")
        print("-" * 55)
        for metric, label in [("mean_ms", "Mean"), ("median_ms", "Median"),
                              ("min_ms", "Min"), ("max_ms", "Max"),
                              ("p95_ms", "P95"), ("p99_ms", "P99")]:
            pt_val = pytorch_stats[metric]
            onnx_val = onnx_stats[metric]
            speedup = pt_val / onnx_val if onnx_val > 0 else 0
            print(f"{label:<12} {pt_val:<15.3f} {onnx_val:<15.3f} {speedup:.2f}x")

    pt_mean = pytorch_stats["mean_ms"]
    onnx_mean = onnx_stats["mean_ms"]
    ts_mean = torchscript_stats["mean_ms"] if torchscript_stats else None

    print("\n" + "-" * 75)
    print("Summary:")
    if ts_mean:
        ts_speedup = pt_mean / ts_mean if ts_mean > 0 else 0
        if ts_speedup > 1:
            print(f"TorchScript is {ts_speedup:.2f}x faster than PyTorch")
        else:
            print(f"PyTorch is {1/ts_speedup:.2f}x faster than TorchScript")
    onnx_speedup = pt_mean / onnx_mean if onnx_mean > 0 else 0
    if onnx_speedup > 1:
        print(f"ONNX is {onnx_speedup:.2f}x faster than PyTorch")
    else:
        print(f"  ⚠️  PyTorch is {1/onnx_speedup:.2f}x faster than ONNX")

    # Find fastest
    if ts_mean:
        fastest = min([("PyTorch", pt_mean), ("TorchScript", ts_mean), ("ONNX", onnx_mean)], key=lambda x: x[1])
        print(f"  ⭐ Fastest: {fastest[0]} ({fastest[1]:.2f}ms)")

    # Real-time check
    target = 10.0  # 10ms for 100Hz
    print(f"\nReal-time feasibility (target: <{target}ms for 100Hz):")
    print(f"  PyTorch:     {'✅' if pt_mean < target else '❌'} {pt_mean:.2f}ms")
    if ts_mean:
        print(f"  TorchScript: {'✅' if ts_mean < target else '❌'} {ts_mean:.2f}ms")
    print(f"  ONNX:        {'✅' if onnx_mean < target else '❌'} {onnx_mean:.2f}ms")


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

    # Export to ONNX
    print("\nExporting model to ONNX...")
    onnx_path = "model.onnx"
    onnx_program = torch.onnx.export(model, input_data, dynamo=True)
    onnx_program.save(onnx_path)
    onnx.checker.check_model(onnx_path)
    print(f"✅ ONNX export successful: {onnx_path}")

    # Create ONNX session
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"]
    )

    # Verify outputs match
    print("\nVerifying ONNX output matches PyTorch output...")
    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(None, {input_name: input_data.numpy()})[0]
    try:
        torch.testing.assert_close(predictions, torch.tensor(onnx_output), rtol=1e-3, atol=1e-4)
        print("✅ Outputs match within tolerance")
    except AssertionError as e:
        print(f"❌ Outputs differ: {e}")

    # Export to TorchScript
    print("\nExporting model to TorchScript...")
    torchscript_path = "model_traced.pt"
    scripted_model = torch.jit.trace(model, input_data)
    scripted_model.save(torchscript_path)
    print(f"✅ TorchScript export successful: {torchscript_path}")

    # Verify TorchScript output
    print("\nVerifying TorchScript output matches PyTorch output...")
    with torch.no_grad():
        ts_output = scripted_model(input_data)
    try:
        torch.testing.assert_close(predictions, ts_output, rtol=1e-5, atol=1e-6)
        print("✅ TorchScript outputs match within tolerance")
    except AssertionError as e:
        print(f"❌ TorchScript outputs differ: {e}")

    if args.benchmark:
        # Benchmark all three
        pytorch_stats = benchmark_pytorch(model, input_data, num_runs=args.num_runs, device=args.device)
        torchscript_stats = benchmark_torchscript(scripted_model, input_data, num_runs=args.num_runs, device=args.device)
        onnx_stats = benchmark_onnx(ort_session, input_data, num_runs=args.num_runs)
        print_comparison(pytorch_stats, onnx_stats, torchscript_stats)



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
