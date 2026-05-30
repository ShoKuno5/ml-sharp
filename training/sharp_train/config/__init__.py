"""Training configuration (dataclasses + YAML load/dump)."""

from .configs import TrainConfig, load_config, save_config

__all__ = ["TrainConfig", "load_config", "save_config"]
