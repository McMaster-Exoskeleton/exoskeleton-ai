.PHONY: help install install-dev test lint format clean build docs

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip3 install -e .

install-dev:  ## Install package with development dependencies
	pip3 install -e ".[dev]"
	pre-commit install

install-all:  ## Install all optional dependencies
	pip3 install -e ".[all]"
	pre-commit install

test:  ## Run tests with pytest
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov --cov-report=html --cov-report=term

test-fast:  ## Run tests excluding slow tests
	pytest -m "not slow"

lint:  ## Run linters (ruff, mypy)
	ruff check src/ tests/
	mypy src/

lint-fix:  ## Fix linting issues automatically
	ruff check --fix src/ tests/

format:  ## Format code with black and ruff
	black src/ tests/
	ruff check --fix src/ tests/

format-check:  ## Check code formatting without modifying files
	black --check src/ tests/
	ruff check src/ tests/

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean:  ## Remove build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

train:  ## Run training script
	python3 scripts/train.py
