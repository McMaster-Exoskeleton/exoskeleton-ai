# Setup Guide

This guide will walk you through setting up the Exoskeleton AI project for development.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- Git
- pip (comes with Python)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/exoskeleton-ai.git
cd exoskeleton-ai
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode with all dev dependencies
make install-dev
```

This will install:
- All core dependencies (PyTorch, NumPy, etc.)
- Development tools (pytest, black, ruff, mypy)
- The package in editable mode (changes reflect immediately)

### 4. Verify Installation

```bash
# Check that the package is installed
python -c "import exoskeleton_ml; print(exoskeleton_ml.__version__)"

# Run tests to ensure everything works
make test

# Or manually:
pytest
```

If all tests pass, you're ready to go!

## Common Issues

### PyTorch Installation

If you need GPU support or have issues with PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For macOS with Apple Silicon
pip install torch torchvision
```

### Missing System Dependencies

On some systems, you might need additional packages:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

#### macOS
```bash
# Install Xcode Command Line Tools if needed
xcode-select --install
```

### Permission Errors

If you get permission errors during installation:

```bash
# Don't use sudo! Instead, use the --user flag
pip install --user -e ".[dev]"

# Or create a virtual environment (recommended)
```

## Next Steps

After setup:

1. Read the [README.md](../README.md) for project overview
2. Check [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
3. Explore the example notebooks in `notebooks/`
4. Try running the example scripts in `scripts/`

## Quick Test Run

Try this quick test to ensure everything is working:

```python
# test_setup.py
from exoskeleton_ml.utils.device import get_device, print_device_info
from exoskeleton_ml.models.baseline import BaselineModel
import torch

# Check device
print_device_info()
device = get_device()

# Create a simple model
model = BaselineModel(input_size=10, hidden_size=32, num_classes=5)
model = model.to(device)

# Test forward pass
x = torch.randn(4, 100, 10).to(device)
y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print("✓ Everything works!")
```

Run it:
```bash
python test_setup.py
```

## Development Workflow

Once set up, your typical workflow will be:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Create a feature branch
git checkout -b feature/my-feature

# 3. Make changes and test
make test
make lint
make format

# 4. Commit (pre-commit hooks will run automatically)
git commit -m "feat: add my feature"

# 5. Push and create PR
git push origin feature/my-feature
```

## Additional Tools (Optional)

### Jupyter

For interactive development:

```bash
# Already installed with dev dependencies
jupyter notebook
# Or
jupyter lab
```

### Weights & Biases

For experiment tracking:

```bash
wandb login
```

### DVC (Data Version Control)

For data versioning (optional):

```bash
pip install -e ".[dvc]"
dvc init
```

## Updating Dependencies

When dependencies are updated in `pyproject.toml`:

```bash
# Reinstall package
pip install -e ".[dev]" --upgrade

# Or use make
make install-dev
```

## Troubleshooting

### Tests Failing

If tests fail after setup:

1. Ensure virtual environment is activated
2. Check Python version: `python --version` (should be 3.10+)
3. Reinstall dependencies: `pip install -e ".[dev]" --force-reinstall`
4. Check for error messages in test output

### Import Errors

If you get import errors:

1. Ensure package is installed in editable mode: `pip install -e .`
2. Check that you're in the correct directory
3. Verify virtual environment is activated

### Pre-commit Hooks Not Running

If pre-commit hooks aren't working:

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Test manually
pre-commit run --all-files
```

## Getting Help

- Open an issue on GitHub
- Check existing documentation in `docs/`
- Ask in project discussions

Happy coding!
