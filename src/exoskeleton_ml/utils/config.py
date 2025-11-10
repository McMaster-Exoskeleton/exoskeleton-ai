"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration as OmegaConf DictConfig object.
    """
    ...


def save_config(config: DictConfig, save_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        save_path: Path where to save the configuration.
    """
    ...


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.

    Args:
        *configs: Variable number of configurations to merge.

    Returns:
        Merged configuration.
    """
    ...


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Configuration as dictionary.
    """
    ...
