"""
Test script for TCN model implementation.

This script tests the TCN model with dummy data to ensure:
1. Model can be instantiated correctly
2. Forward pass works with expected input shapes
3. Backward pass computes gradients
4. Model outputs correct shapes
5. Model factory integration works

Usage:
    python scripts/test_tcn_model.py
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exoskeleton_ml.models import TCN, create_model


def test_tcn_forward():
    """Test TCN forward pass with dummy data."""
    print("=" * 80)
    print("Test 1: TCN Forward Pass")
    print("=" * 80)

    # Model parameters
    input_size = 28
    output_size = 4
    num_channels = [25, 25, 25, 25, 25]
    kernel_size = 7
    dropout = 0.2
    eff_hist = (kernel_size - 1) * (2 ** len(num_channels) - 1) + 1

    # Create model
    model = TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        eff_hist=eff_hist,
        spatial_dropout=False,
        activation="ReLU",
        norm="weight_norm",
    )

    print(f"✅ Model created successfully")
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Effective history: {model.get_effective_history()} timesteps")

    # Create dummy input
    batch_size = 4
    seq_len = 200
    x = torch.randn(batch_size, seq_len, input_size)

    print(f"\n✅ Dummy input created: {x.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {output_size})")

    # Verify output shape
    assert output.shape == (batch_size, seq_len, output_size), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {seq_len}, {output_size})"

    print("\n✅ Test 1 PASSED: Forward pass works correctly\n")


def test_tcn_backward():
    """Test TCN backward pass (gradient computation)."""
    print("=" * 80)
    print("Test 2: TCN Backward Pass")
    print("=" * 80)

    # Create model
    model = TCN(
        input_size=28,
        output_size=4,
        num_channels=[25, 25, 25],
        kernel_size=5,
        dropout=0.2,
        eff_hist=61,
    )

    # Create dummy data
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 28)
    target = torch.randn(batch_size, seq_len, 4)

    # Forward pass
    model.train()
    output = model(x)

    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)

    print(f"✅ Forward pass successful, loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print(f"✅ Backward pass successful")

    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed!"

    print(f"✅ Gradients computed for model parameters")

    # Check gradient magnitudes
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    print(f"✅ Total gradient norm: {total_grad_norm:.4f}")

    print("\n✅ Test 2 PASSED: Backward pass works correctly\n")


def test_model_factory():
    """Test model creation via factory."""
    print("=" * 80)
    print("Test 3: Model Factory")
    print("=" * 80)

    # Load TCN config
    config_path = Path(__file__).parent.parent / "configs" / "model" / "tcn.yaml"
    config = OmegaConf.load(config_path)

    print(f"✅ Config loaded from {config_path}")

    # Create model via factory
    model = create_model(config)

    print(f"✅ Model created via factory")
    print(f"   Type: {type(model).__name__}")
    print(f"   Parameters: {model.get_num_parameters():,}")

    # Test forward pass
    x = torch.randn(2, 100, 28)
    output = model(x)

    print(f"✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")

    assert output.shape == (2, 100, 4), f"Output shape mismatch! Got {output.shape}"

    print("\n✅ Test 3 PASSED: Model factory works correctly\n")


def test_variable_length_sequences():
    """Test TCN with variable-length sequences (simulating padding)."""
    print("=" * 80)
    print("Test 4: Variable-Length Sequences")
    print("=" * 80)

    model = TCN(
        input_size=28,
        output_size=4,
        num_channels=[25, 25, 25],
        kernel_size=5,
        dropout=0.2,
        eff_hist=61,
    )

    # Create batch with different sequence lengths
    batch_size = 3
    max_seq_len = 150
    actual_lengths = [100, 120, 150]

    # Create padded input
    x = torch.zeros(batch_size, max_seq_len, 28)
    for i, length in enumerate(actual_lengths):
        x[i, :length, :] = torch.randn(length, 28)

    # Create mask
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(actual_lengths):
        mask[i, :length] = True

    print(f"✅ Variable-length batch created")
    print(f"   Lengths: {actual_lengths}")
    print(f"   Max length: {max_seq_len}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")

    # Verify non-padded regions have non-zero values
    for i, length in enumerate(actual_lengths):
        non_zero = (output[i, :length, :].abs() > 1e-6).any()
        assert non_zero, f"Sample {i}: Non-padded region has all zeros!"

    print(f"✅ Non-padded regions have valid predictions")

    print("\n✅ Test 4 PASSED: Variable-length sequences work correctly\n")


def test_different_model_sizes():
    """Test different TCN model sizes (small, medium, large)."""
    print("=" * 80)
    print("Test 5: Different Model Sizes")
    print("=" * 80)

    configs = {
        "small": {
            "num_channels": [15, 15, 15, 15],
            "kernel_size": 5,
        },
        "medium": {
            "num_channels": [25, 25, 25, 25, 25],
            "kernel_size": 7,
        },
        "large": {
            "num_channels": [50, 50, 50, 50, 50, 50],
            "kernel_size": 9,
        },
    }

    for name, cfg in configs.items():
        num_levels = len(cfg["num_channels"])
        eff_hist = (cfg["kernel_size"] - 1) * (2 ** num_levels - 1) + 1

        model = TCN(
            input_size=28,
            output_size=4,
            num_channels=cfg["num_channels"],
            kernel_size=cfg["kernel_size"],
            dropout=0.2,
            eff_hist=eff_hist,
        )

        num_params = model.get_num_parameters()
        print(f"\n{name.upper()} model:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Effective history: {eff_hist} timesteps ({eff_hist/100:.2f}s)")

        # Test forward pass
        x = torch.randn(2, 100, 28)
        output = model(x)

        assert output.shape == (2, 100, 4), f"{name} model: output shape mismatch"
        print(f"  ✅ Forward pass successful")

    print("\n✅ Test 5 PASSED: All model sizes work correctly\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TCN Model Tests")
    print("=" * 80 + "\n")

    try:
        test_tcn_forward()
        test_tcn_backward()
        test_model_factory()
        test_variable_length_sequences()
        test_different_model_sizes()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe TCN model is ready for training!")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
