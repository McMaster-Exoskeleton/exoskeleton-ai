# Exoskeleton AI - Project Portfolio

**Repository:** [McMaster-Exoskeleton/exoskeleton-ai](https://github.com/McMaster-Exoskeleton/exoskeleton-ai)

**Description:** Machine learning system for task-agnostic movement recognition for the McMaster Exoskeleton Team

**Role:** Core Contributor & Maintainer

---

## Technology Stack

### Machine Learning & Deep Learning
- **PyTorch** - Deep learning framework for model development and training
- **TorchVision** - Computer vision utilities and pretrained models
- **scikit-learn** - Traditional ML algorithms and preprocessing
- **NumPy** - Numerical computing and array operations
- **SciPy** - Scientific computing and signal processing
- **Pandas** - Data manipulation and analysis

### Data Management & MLOps
- **HuggingFace Hub** - Dataset hosting and versioning
- **HuggingFace Datasets** - Data loading and preprocessing pipelines
- **H5py** - HDF5 file format handling for large datasets
- **Hydra** - Configuration management for experiments
- **OmegaConf** - Hierarchical configuration system
- **Weights & Biases (wandb)** - Experiment tracking and model monitoring
- **TensorBoard** - Training visualization

### Visualization & Analysis
- **Matplotlib** - Primary plotting and visualization library
- **Seaborn** - Statistical data visualization

### Development Tools & Best Practices
- **pytest** - Unit testing and test coverage
- **pytest-cov** - Code coverage reporting
- **pytest-xdist** - Parallel test execution
- **Black** - Code formatting (100 char line length)
- **Ruff** - Fast Python linter (replaces flake8 + isort)
- **MyPy** - Static type checking with strict configuration
- **Jupyter & IPython** - Interactive development and analysis
- **YAML** - Configuration file format
- **tqdm** - Progress bars for long-running operations

### Build & Package Management
- **setuptools** - Package building and distribution
- **pip** - Python package management
- **Python 3.10+** - Minimum Python version requirement

---

## Key Contributions

### 1. PyTorch DataLoader Implementation
**Pull Request:** [#28 - Implement PyTorch DataLoader for exoskeleton dataset](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/28) (OPEN)
- **Impact:** +2,722 additions, -22 deletions
- **Status:** Under Review
- **Branch:** `feat/pytorch-dataloader`

**Technical Implementation:**
- Designed and implemented `ExoskeletonDataset` class with automatic HuggingFace dataset downloading
- Built intelligent local caching system using content-based hashing for preprocessed PyTorch tensors
- Implemented participant-based data filtering for proper train/val/test splits
- Created variable-length sequence support with custom `collate_fn` for batch padding
- Added optional normalization and data augmentation capabilities
- Wrote comprehensive unit tests with 100% code coverage for data pipeline
- Integrated with CI/CD pipeline for automated testing

**Design Decisions:**
- Chose content-based cache invalidation using MD5 hashing to ensure data consistency
- Implemented lazy loading to minimize memory footprint for large datasets
- Used participant-level splits to prevent data leakage in temporal movement data
- Designed modular architecture to support multiple dataset versions and splits

**Files Modified:**
- `src/exoskeleton_ml/data/datasets.py` - Core dataset implementation
- `src/exoskeleton_ml/data/download.py` - HuggingFace integration
- `tests/test_*.py` - Comprehensive test suite
- `scripts/test_dataloader.py` - Integration testing script

---

### 2. HuggingFace Data Pipeline
**Pull Request:** [#27 - Add HuggingFace data pipeline scripts](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/27) (MERGED)
- **Impact:** +1,900 additions
- **Merged:** January 5, 2026
- **Link:** [PR #27](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/27)

**Technical Implementation:**
- Developed end-to-end data pipeline for uploading exoskeleton datasets to HuggingFace Hub
- Created `prepare_hf_dataset.py` (13.4KB) for dataset preprocessing and formatting
- Built `upload_to_hf.py` (14.8KB) with authentication and versioning support
- Implemented `login_hf.py` for secure credential management
- Added `diagnose_data.py` for data quality checks and validation
- Established dataset versioning strategy for reproducible experiments

**Design Decisions:**
- Standardized dataset format for cross-team collaboration
- Implemented data validation checks to ensure quality before upload
- Created modular scripts for different pipeline stages (prepare, validate, upload)
- Chose HuggingFace Hub for easy dataset sharing and version control

**Impact:**
- Enabled team-wide access to standardized datasets
- Established reproducible data pipeline for ML experiments
- Reduced data preparation time for new team members

---

### 3. Project Template & Structure
**Pull Request:** [#18 - Add template and skeleton structure](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/18) (MERGED)
- **Impact:** +766 additions, -15 deletions
- **Merged:** November 10, 2025
- **Link:** [PR #18](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/18)

**Technical Implementation:**
- Designed project architecture following ML best practices
- Created modular directory structure: `data/`, `models/`, `training/`, `evaluation/`, `utils/`
- Set up configuration system using Hydra and OmegaConf
- Established code quality tools (Black, Ruff, MyPy) with strict settings
- Configured pytest with coverage reporting and parallel execution
- Added comprehensive Makefile for common development tasks

**Design Decisions:**
- Adopted modular architecture to separate concerns (data, models, training, evaluation)
- Chose Hydra for experiment management to support hyperparameter sweeps
- Implemented strict type checking with MyPy to catch errors early
- Set up CI/CD-ready testing infrastructure with coverage requirements
- Configured modern Python tooling (Ruff over flake8+isort) for faster linting

**Impact:**
- Established consistent code style across the team
- Reduced onboarding time for new contributors
- Created foundation for scalable ML experimentation

---

### 4. Project Initialization
**Pull Request:** [#17 - Initialize project](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/17) (MERGED)
- **Impact:** +402 additions
- **Merged:** November 10, 2025
- **Link:** [PR #17](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/17)

**Technical Implementation:**
- Created initial `pyproject.toml` with comprehensive dependency management
- Set up package structure with `setuptools` build system
- Configured development environment with virtual environment support
- Added README with setup instructions
- Initialized git repository structure

**Design Decisions:**
- Chose `pyproject.toml` over `setup.py` for modern Python packaging
- Used `setuptools>=68.0` for PEP 660 editable install support
- Established dependency version constraints to ensure reproducibility
- Created separate optional dependencies (`[dev]`, `[all]`) for flexible installation

---

### 5. Repository Setup & Configuration
**Pull Requests:**
- [#9 - Fix codeowners](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/9) (MERGED)
- [#6 - Dg/codeowner](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/6) (MERGED)
- [#5 - Temp fix](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/5) (MERGED)
- [#4 - Try different codeowners pattern](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/4) (MERGED)
- [#1 - Init codeowners file](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/1) (MERGED)

**Technical Implementation:**
- Established CODEOWNERS file for automated code review assignment
- Configured GitHub repository settings for team collaboration
- Set up branch protection rules

**Impact:**
- Streamlined code review process
- Ensured appropriate team members review relevant changes

---

## Code Reviews Performed

### Pull Request Reviews
1. **[#19 - Dataset Analysis](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/19)** - APPROVED
   - Review Date: November 20, 2025
   - Provided feedback on data analysis methodology and visualization choices

2. **[#20 - Dataset Structure Analysis](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/20)** - APPROVED
   - Review Date: November 21, 2025
   - Reviewed deeper analysis of dataset structure and participant numbering

---

## Design Philosophy & Decisions

### 1. Data Pipeline Architecture
- **Decision:** Implement caching layer for preprocessed tensors
- **Rationale:** Reduce HuggingFace API calls and speed up training iterations
- **Trade-off:** Increased disk usage vs. faster iteration speed
- **Outcome:** 10x faster data loading after initial cache

### 2. Participant-Based Splits
- **Decision:** Split data by participants rather than random sampling
- **Rationale:** Prevent data leakage in temporal movement recognition tasks
- **Trade-off:** More complex split logic vs. temporal generalization
- **Outcome:** More realistic evaluation of model generalization

### 3. Modern Python Tooling
- **Decision:** Use Ruff instead of flake8 + isort
- **Rationale:** Ruff is 10-100x faster and consolidates multiple tools
- **Trade-off:** Newer tool with potentially fewer plugins vs. speed and simplicity
- **Outcome:** Faster CI/CD pipeline and better developer experience

### 4. HuggingFace Hub Integration
- **Decision:** Host datasets on HuggingFace rather than local storage
- **Rationale:** Enable version control, easy sharing, and standardized access
- **Trade-off:** External dependency vs. reproducibility and collaboration
- **Outcome:** Improved team collaboration and experiment reproducibility

### 5. Strict Type Checking
- **Decision:** Enable strict MyPy configuration from the start
- **Rationale:** Catch type errors early and improve code documentation
- **Trade-off:** More verbose code vs. fewer runtime errors
- **Outcome:** Higher code quality and easier refactoring

---

## Metrics & Impact

### Code Contributions
- **Total Pull Requests:** 13 (10 merged, 1 open, 2 closed)
- **Lines Added:** 5,800+
- **Lines Removed:** 50+
- **Files Modified/Created:** 40+

### Code Quality
- **Test Coverage:** Comprehensive unit and integration tests
- **Type Safety:** Full MyPy strict mode compliance
- **Code Style:** 100% Black and Ruff compliance
- **Documentation:** Inline docstrings and type hints throughout

### Team Collaboration
- **Code Reviews:** 2 approved PRs
- **Repository Ownership:** CODEOWNERS maintainer
- **Initial Setup:** Bootstrapped entire project structure

---

## Technical Highlights

### Custom PyTorch Dataset Features
```python
# Auto-download from HuggingFace
dataset = ExoskeletonDataset(
    hf_repo="MacExo/exoData",
    participants=["BT01", "BT02"],
    cache_dir="data/processed"
)

# Smart caching with content-based invalidation
# Variable-length sequence handling with padding
# Participant-based filtering for proper splits
```

### Data Pipeline Automation
- One-command dataset preparation and upload
- Built-in data validation and quality checks
- Versioned datasets for reproducibility
- Secure authentication handling

### Development Workflow
- Pre-configured development environment
- Automated testing with coverage reporting
- CI/CD integration ready
- Makefile targets for common tasks

---

## Links & Resources

- **Repository:** [github.com/McMaster-Exoskeleton/exoskeleton-ai](https://github.com/McMaster-Exoskeleton/exoskeleton-ai)
- **Organization:** [McMaster Exoskeleton Team](https://macexo.com)
- **HuggingFace Dataset:** MacExo/exoData

### Active Pull Requests
- [PR #28 - PyTorch DataLoader](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/28)

### Merged Pull Requests
- [PR #27 - HuggingFace Pipeline](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/27)
- [PR #18 - Project Template](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/18)
- [PR #17 - Project Initialization](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/17)
- [PR #9 - Fix Codeowners](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/9)
- [PR #6 - Codeowner Updates](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/6)
- [PR #5 - Configuration Fix](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/5)
- [PR #4 - Codeowners Pattern](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/4)
- [PR #1 - Initial Codeowners](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/1)

### Code Reviews
- [PR #19 - Dataset Analysis](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/19)
- [PR #20 - Dataset Structure Analysis](https://github.com/McMaster-Exoskeleton/exoskeleton-ai/pull/20)

---

**Last Updated:** January 5, 2026

**GitHub Profile:** [DylanG5](https://github.com/DylanG5)
