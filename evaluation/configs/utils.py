"""Loaders for query-generation and algorithm config files."""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any

from compaction.query_generation import QueryConfig


def _load_module(config_path: Path):
    """Load a Python module from a file path."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config_module'] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def load_query_config(config_path: str = 'repeat') -> QueryConfig:
    """Load `config` from evaluation/configs/query_generation/<config_path>.py (a bare name is allowed)."""
    if not config_path.startswith('evaluation/configs/'):
        if not config_path.endswith('.py'):
            config_path = f"{config_path}.py"
        config_path = f"evaluation/configs/query_generation/{config_path}"

    config_module = _load_module(Path(config_path))

    if not hasattr(config_module, 'config'):
        raise AttributeError(
            f"Config file {config_path} must define a 'config' variable"
        )

    return config_module.config


def load_algorithm_config(config_path: str = 'fast', target_size: int = None) -> Dict[str, Dict[str, Any]]:
    """Load {method_name: kwargs} from evaluation/configs/algorithms/<config_path>.py.

    Uses `get_config(target_size=...)` if the file defines it, else `config`.
    """
    if not config_path.startswith('evaluation/configs/'):
        if not config_path.endswith('.py'):
            config_path = f"{config_path}.py"
        config_path = f"evaluation/configs/algorithms/{config_path}"

    config_module = _load_module(Path(config_path))

    if hasattr(config_module, 'get_config') and callable(config_module.get_config):
        return config_module.get_config(target_size=target_size)

    if not hasattr(config_module, 'config'):
        raise AttributeError(
            f"Config file {config_path} must define a 'config' variable or 'get_config' function"
        )

    return config_module.config


def get_method_kwargs(config: Dict[str, Dict[str, Any]], method_name: str) -> Dict[str, Any]:
    """Hyperparameters for `method_name`, or {} if absent."""
    return config.get(method_name, {})
