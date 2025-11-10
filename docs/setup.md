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
