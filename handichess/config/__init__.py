"""Configuration loading and pattern definitions."""

import os
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from the config directory."""
    filepath = CONFIG_DIR / filename
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def load_patterns() -> dict:
    """Load removal pattern configurations."""
    return load_yaml("patterns.yaml")


def load_defaults() -> dict:
    """Load default hyperparameters."""
    return load_yaml("default.yaml")
