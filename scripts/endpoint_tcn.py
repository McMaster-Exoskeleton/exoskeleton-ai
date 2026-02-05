"""
ONNX-only inference script for Raspberry Pi deployment. This script is for using

Usage:
  python scripts/endpoint_tcn.py --model-path outputs/model.onnx

"""

import onnxruntime
import numpy as np
import argparse
from pathlib import Path

def prediction():
    parser = argparse.ArgumentParser(description="Simple ONNX Inference")
    parser.add_argument("--model-path", type=str, default="model.onnx", help="Path to .onnx file")
    args = parser.parse_args()

    # 1. Load the ONNX model
    if not Path(args.model_path).exists():
        print(f"Error: Model file '{args.model_path}' not found.")
        return

    # Use CPU providers (standard for Raspberry Pi)
    session = onnxruntime.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])

    # 2. Prepare one sample (Batch=1, Sequence=100, Features=28)
    
    # Match the shape used during your export

    ##CHANGE TO INPUT
    input_shape = (1, 100, 28)
    sample_input = np.random.randn(*input_shape).astype(np.float32)

    # 3. Run Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample_input})

    # 4. Display Result
    predictions = outputs[0]
    print(f"Successfully ran inference on {args.model_path}")
    print(f"Input Shape:  {input_shape}")
    print(f"Output Shape: {predictions.shape}")
    print("\n--- Sample Prediction (First 3 timesteps) ---")
    print(predictions[0, :3, :])

