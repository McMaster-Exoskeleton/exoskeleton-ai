"""
Export a trained PyTorch TCN checkpoint to ONNX format for Raspberry Pi deployment.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model-path outputs/2026-01-08/18-29-05/best_model.pt
    python scripts/export_onnx.py --output outputs/model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from omegaconf import OmegaConf

from exoskeleton_ml.models import create_model

DEFAULT_MODEL_PATH = "outputs/2026-01-08/18-29-05/best_model.pt"


def load_model(model_path: Path) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(model_path, map_location="cpu")

    config_path = model_path.parent / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config not found at {config_path}")

    config = OmegaConf.load(config_path)
    model = create_model(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    window_size: int = config.model.effective_history
    return model, window_size


def export(model_path: Path, output_path: Path) -> None:
    print(f"Loading model: {model_path}")
    model, window_size = load_model(model_path)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:     {num_params:,}")
    print(f"  Window size:    {window_size} timesteps")
    print(f"  Input shape:    (1, {window_size}, 28)")
    print(f"  Output shape:   (1, {window_size}, 4)")

    # Representative input for tracing
    dummy_input = torch.randn(1, window_size, 28)

    with torch.no_grad():
        pytorch_output = model(dummy_input)

    print(f"\nExporting to ONNX: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_program = torch.onnx.export(model, dummy_input, dynamo=True)
    onnx_program.save(str(output_path))

    print("Checking ONNX model...")
    onnx.checker.check_model(str(output_path))
    print("  ONNX check passed.")

    print("\nVerifying outputs match PyTorch...")
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(
        str(output_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(None, {input_name: dummy_input.numpy()})[0]

    try:
        torch.testing.assert_close(
            pytorch_output,
            torch.tensor(onnx_output),
            rtol=1e-3,
            atol=1e-4,
        )
        print("  Outputs match within tolerance. Export verified.")
    except AssertionError as e:
        print(f"  WARNING: outputs differ: {e}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone. Saved to: {output_path}  ({size_mb:.2f} MB)")
    print(f"\nSCP to Pi:")
    print(f"  scp {output_path} pi@<PI_IP>:~/exoskeleton-ai/outputs/model.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TCN checkpoint to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to best_model.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/model.onnx",
        help="Output path for the .onnx file",
    )
    args = parser.parse_args()

    export(Path(args.model_path), Path(args.output))


if __name__ == "__main__":
    main()
