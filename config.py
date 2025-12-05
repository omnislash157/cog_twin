"""
Configuration loader for COG_TWIN.

Single source of truth. One file, one loader.

Usage:
    from config import get_config, cfg

    # Full config dict
    config = get_config()

    # Dotted access helper
    model_name = cfg("model.name")
    top_k = cfg("retrieval.process_top_k", default=10)

Version: 1.0.0
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# Load .env file to make environment variables available
load_dotenv()

logger = logging.getLogger(__name__)

# Cached config
_config: Optional[dict] = None
_config_path: Optional[Path] = None


def load_config(path: Optional[Path] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        path: Path to config.yaml. If None, searches:
              1. Current directory
              2. Project root (where this file lives)
              3. Parent directories

    Returns:
        Configuration dict (empty if no config found)
    """
    global _config, _config_path

    # Return cached if already loaded from same path
    if _config is not None and (path is None or path == _config_path):
        return _config

    # Find config file
    if path is None:
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]
        for p in search_paths:
            if p.exists():
                path = p
                break

    if path is None or not path.exists():
        logger.warning("No config.yaml found, using defaults")
        _config = {}
        return _config

    # Load YAML
    with open(path) as f:
        _config = yaml.safe_load(f) or {}

    _config_path = path
    logger.info(f"Loaded config from {path}")

    return _config


def get_config() -> dict:
    """Get the full configuration dict."""
    if _config is None:
        load_config()
    return _config


def cfg(key: str, default: Any = None) -> Any:
    """
    Get a config value by dotted key path.

    Args:
        key: Dotted path like "model.name" or "retrieval.process_top_k"
        default: Value to return if key not found

    Returns:
        Config value or default

    Example:
        model = cfg("model.name", "claude-sonnet-4-20250514")
        top_k = cfg("retrieval.process_top_k", 10)
    """
    config = get_config()

    # Navigate dotted path
    parts = key.split(".")
    value = config

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default

    return value


def reload_config(path: Optional[Path] = None) -> dict:
    """Force reload configuration from disk."""
    global _config
    _config = None
    return load_config(path)


# =============================================================================
# CONVENIENCE ACCESSORS
# =============================================================================

def get_model_config() -> dict:
    """Get model configuration section."""
    return get_config().get("model", {})


def get_retrieval_config() -> dict:
    """Get retrieval configuration section."""
    return get_config().get("retrieval", {})


def get_cognitive_config() -> dict:
    """Get cognitive system configuration section."""
    return get_config().get("cognitive", {})


def get_paths() -> dict:
    """Get paths configuration section."""
    return get_config().get("paths", {})


def get_data_dir() -> Path:
    """Get data directory path (resolved relative to project root)."""
    data_dir = cfg("paths.data_dir", "./data")
    path = Path(data_dir)

    # If relative, resolve relative to this config.py file (project root)
    if not path.is_absolute():
        project_root = Path(__file__).parent
        path = project_root / path

    return path.resolve()


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging from config.yaml settings."""
    log_config = get_config().get("logging", {})

    level = log_config.get("level", "INFO")
    fmt = log_config.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )


# =============================================================================
# ENV VAR OVERRIDES
# =============================================================================

def get_api_key(provider: str = "anthropic") -> Optional[str]:
    """
    Get API key with env var override.

    Priority:
        1. Environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY)
        2. Config file (not recommended for secrets)
    """
    env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "xai": "XAI_API_KEY",
    }

    env_var = env_vars.get(provider.lower())
    if env_var:
        return os.getenv(env_var)

    return None
