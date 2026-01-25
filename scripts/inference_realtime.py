"""
Real-time inference script for TCN model on streaming exoskeleton data.

This demonstrates how to use the trained TCN model for real-time
exoskeleton control with streaming sensor data.

Usage:
    # Simulated streaming data
    python scripts/inference_realtime.py --model outputs/.../best_model.pt

    # Real hardware (future)
    python scripts/inference_realtime.py --model best_model.pt --hardware
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exoskeleton_ml.models import TCN
from exoskeleton_ml.utils import load_checkpoint


class RealtimePredictor:
    """Real-time moment prediction with streaming data.

    Uses a sliding buffer approach to maintain temporal context
    while processing incoming sensor data.
    """

    def __init__(
        self,
        model: nn.Module,
        effective_history: int = 187,
        device: str = "cpu",
        normalization_stats: Optional[dict] = None,
    ):
        """Initialize real-time predictor.

        Args:
            model: Trained TCN model.
            effective_history: Model's receptive field (timesteps).
            device: Device to run inference on.
            normalization_stats: Mean/std for input normalization.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.eff_hist = effective_history

        # Circular buffer for maintaining history
        self.buffer = deque(maxlen=effective_history)

        # Normalization stats (if using)
        self.norm_stats = normalization_stats

        # Warmup flag
        self.warmed_up = False

    def warmup(self, initial_data: torch.Tensor) -> None:
        """Warm up the buffer with initial data.

        Args:
            initial_data: Initial sensor readings (eff_hist, 28).
        """
        assert initial_data.shape[0] >= self.eff_hist, \
            f"Need at least {self.eff_hist} timesteps for warmup"

        # Fill buffer
        for t in range(self.eff_hist):
            self.buffer.append(initial_data[t].cpu().numpy())

        self.warmed_up = True
        print(f"✅ Warmed up with {self.eff_hist} timesteps")

    def predict_step(self, new_data: np.ndarray) -> np.ndarray:
        """Predict joint moments for a single new timestep.

        Args:
            new_data: New sensor reading (28,) or (n, 28).

        Returns:
            Predicted joint moments (4,) or (n, 4).
        """
        if not self.warmed_up:
            raise RuntimeError("Must call warmup() before predict_step()")

        # Handle single timestep or batch
        single_step = (new_data.ndim == 1)
        if single_step:
            new_data = new_data[np.newaxis, :]  # (1, 28)

        predictions = []

        for timestep in new_data:
            # Add to buffer
            self.buffer.append(timestep)

            # Convert buffer to tensor
            sequence = torch.tensor(
                np.array(self.buffer),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # (1, eff_hist, 28)

            # Normalize if needed
            if self.norm_stats is not None:
                mean = torch.tensor(
                    self.norm_stats['mean'],
                    dtype=torch.float32,
                    device=self.device
                )
                std = torch.tensor(
                    self.norm_stats['std'],
                    dtype=torch.float32,
                    device=self.device
                )
                sequence = (sequence - mean) / std

            # Predict
            with torch.no_grad():
                output = self.model(sequence)  # (1, eff_hist, 4)
                moment = output[0, -1, :].cpu().numpy()  # Last timestep (4,)
                predictions.append(moment)

        predictions = np.array(predictions)
        return predictions[0] if single_step else predictions

    def predict_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Predict for a chunk of data (more efficient than step-by-step).

        Args:
            chunk: Chunk of sensor data (chunk_size, 28).

        Returns:
            Predicted moments (chunk_size, 4).
        """
        if not self.warmed_up:
            raise RuntimeError("Must call warmup() before predict_chunk()")

        chunk_size = chunk.shape[0]

        # Combine buffer with new chunk
        buffer_array = np.array(self.buffer)  # (eff_hist, 28)
        full_sequence = np.vstack([buffer_array, chunk])  # (eff_hist + chunk_size, 28)

        # Convert to tensor
        sequence = torch.tensor(
            full_sequence,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, total_len, 28)

        # Normalize if needed
        if self.norm_stats is not None:
            mean = torch.tensor(
                self.norm_stats['mean'],
                dtype=torch.float32,
                device=self.device
            )
            std = torch.tensor(
                self.norm_stats['std'],
                dtype=torch.float32,
                device=self.device
            )
            sequence = (sequence - mean) / std

        # Predict entire sequence
        with torch.no_grad():
            output = self.model(sequence)  # (1, total_len, 4)

        # Extract predictions for new chunk only (ignore buffer predictions)
        predictions = output[0, self.eff_hist:, :].cpu().numpy()  # (chunk_size, 4)

        # Update buffer with last eff_hist timesteps
        self.buffer.clear()
        for t in range(-self.eff_hist, 0):
            self.buffer.append(full_sequence[t])

        return predictions

    def reset(self) -> None:
        """Reset the predictor state."""
        self.buffer.clear()
        self.warmed_up = False


def simulate_realtime_streaming(predictor: RealtimePredictor, test_trial: np.ndarray):
    """Simulate real-time streaming with a test trial.

    Args:
        predictor: RealtimePredictor instance.
        test_trial: Full trial data (seq_len, 28).
    """
    print("\n" + "=" * 80)
    print("Simulating Real-Time Streaming")
    print("=" * 80)

    seq_len = test_trial.shape[0]
    eff_hist = predictor.eff_hist

    # Warmup with first eff_hist timesteps
    print(f"\n1. Warming up with first {eff_hist} timesteps...")
    warmup_data = torch.tensor(test_trial[:eff_hist], dtype=torch.float32)
    predictor.warmup(warmup_data)

    # Option 1: Step-by-step prediction (real-time simulation)
    print(f"\n2. Step-by-step prediction (simulating 100Hz streaming)...")
    step_predictions = []

    for t in range(eff_hist, min(eff_hist + 100, seq_len)):  # First 100 steps
        new_sample = test_trial[t]
        moment = predictor.predict_step(new_sample)
        step_predictions.append(moment)

        if (t - eff_hist) % 20 == 0:
            print(f"   t={t}: Hip_L={moment[0]:.3f}, Hip_R={moment[1]:.3f}, "
                  f"Knee_L={moment[2]:.3f}, Knee_R={moment[3]:.3f} Nm/kg")

    step_predictions = np.array(step_predictions)
    print(f"✅ Processed {len(step_predictions)} timesteps step-by-step")

    # Reset for chunk processing
    predictor.reset()
    predictor.warmup(warmup_data)

    # Option 2: Chunk-based prediction (more efficient)
    print(f"\n3. Chunk-based prediction (processing 500 timesteps at a time)...")
    chunk_size = 500
    chunk_predictions = []

    num_chunks = (seq_len - eff_hist) // chunk_size

    for i in range(min(3, num_chunks)):  # Process first 3 chunks
        start = eff_hist + i * chunk_size
        end = start + chunk_size

        chunk = test_trial[start:end]
        moments = predictor.predict_chunk(chunk)
        chunk_predictions.append(moments)

        print(f"   Chunk {i+1}: Processed timesteps {start}-{end}")
        print(f"      Mean moments: Hip_L={moments[:, 0].mean():.3f}, "
              f"Hip_R={moments[:, 1].mean():.3f}, "
              f"Knee_L={moments[:, 2].mean():.3f}, "
              f"Knee_R={moments[:, 3].mean():.3f} Nm/kg")

    chunk_predictions = np.vstack(chunk_predictions)
    print(f"✅ Processed {len(chunk_predictions)} timesteps in chunks")

    # Compare both methods (should be identical for overlapping region)
    overlap_len = min(len(step_predictions), len(chunk_predictions))
    diff = np.abs(step_predictions[:overlap_len] - chunk_predictions[:overlap_len]).mean()
    print(f"\n4. Verification: Average difference between methods: {diff:.6f} (should be ~0)")

    if diff < 1e-5:
        print("   ✅ Both methods produce identical results!")
    else:
        print("   ⚠️  Methods differ - check implementation")


def main():
    """Main function for real-time inference demo."""
    parser = argparse.ArgumentParser(description="Real-time TCN inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")

    args = parser.parse_args()

    print("=" * 80)
    print("Real-Time TCN Inference Demo")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {args.model}...")

    # Create model (using default TCN config)
    model = TCN(
        input_size=28,
        output_size=4,
        num_channels=[25, 25, 25, 25, 25],
        kernel_size=7,
        dropout=0.2,
        eff_hist=187,
    )

    # Load checkpoint
    device = torch.device(args.device)
    checkpoint_info = load_checkpoint(args.model, model, device=str(device))

    print(f"✅ Model loaded (epoch {checkpoint_info['epoch']}, "
          f"loss {checkpoint_info['loss']:.4f})")

    # Create predictor
    predictor = RealtimePredictor(model, effective_history=187, device=str(device))
    print(f"✅ Real-time predictor initialized")

    # Load test data
    print(f"\nLoading test trial for simulation...")
    from exoskeleton_ml.data import ExoskeletonDataset

    test_dataset = ExoskeletonDataset(
        hf_repo="MacExo/exoData",
        participants=["BT15"],  # Test participant
        cache_dir="data/processed/phase1",
        normalize=True,
    )

    # Get first trial
    sample = test_dataset[0]
    test_trial = sample['inputs'].numpy()  # (seq_len, 28)

    print(f"✅ Loaded test trial: {sample['trial_name']}")
    print(f"   Duration: {sample['sequence_length']} timesteps "
          f"({sample['sequence_length']/100:.1f}s)")

    # Run simulation
    simulate_realtime_streaming(predictor, test_trial)

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nFor actual hardware deployment:")
    print("  1. Replace test_trial with real sensor readings")
    print("  2. Call predictor.predict_step() at 100Hz")
    print("  3. Send predictions to exoskeleton controller")
    print("  4. Latency: ~1-5ms per prediction (GPU) or ~10-20ms (CPU)")


if __name__ == "__main__":
    main()
