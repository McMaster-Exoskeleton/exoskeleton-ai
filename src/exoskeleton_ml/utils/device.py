"""Device management utilities for PyTorch."""

import torch


def get_device(device: str | None = None) -> torch.device:
    """Get the appropriate device for computation.

    Args:
        device: Requested device ('cuda', 'cpu', 'mps', or None for auto).

    Returns:
        torch.device object.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device)


def get_device_name(device: torch.device) -> str:
    """Get human-readable device name.

    Args:
        device: Device to get name for.

    Returns:
        Device name string.
    """
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    elif device.type == "mps":
        return "Apple MPS"
    else:
        return "CPU"


def print_device_info() -> None:
    """Print information about available devices."""
    print("Device Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  Current device: {get_device_name(get_device())}")
