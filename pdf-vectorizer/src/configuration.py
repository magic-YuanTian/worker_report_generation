# src/configuration.py
# Configuration loader with environment variable substitution.
#   load_config         – load config/config.yaml with env-var expansion; LRU-cached
#   get_paths_config    – resolve data folder paths (raw, processed)
#   get_extraction_config – get PDF extraction settings
#   get_chunking_config – get text chunking settings
#   get_vectorization_config – get embedding/vectorization settings

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def _project_root() -> Path:
    """Find project root by searching for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: go up two levels from this file
    return current.parents[2]


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in config values.
    
    Syntax: ${VAR_NAME:default_value} or ${VAR_NAME}
    
    Examples:
        ${OPENAI_API_KEY:sk-default}  → uses OPENAI_API_KEY or 'sk-default'
        ${EMBEDDING_MODEL:all-MiniLM-L6-v2}  → uses EMBEDDING_MODEL or the default
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.getenv(var_name, default)
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


@lru_cache(maxsize=1)
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the pipeline configuration with env-var substitution."""
    load_dotenv()  # Load .env file if present
    
    path = config_path or (_project_root() / "config" / "config.yaml")
    path = path.resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    
    return _substitute_env_vars(config)


def get_paths_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Get resolved data folder paths."""
    cfg = config or load_config()
    data_cfg = cfg.get("data", {})
    
    root = _project_root()
    paths = {
        "raw": (root / data_cfg.get("raw", "data/raw")).resolve(),
        "processed": (root / data_cfg.get("processed", "data/processed")).resolve(),
    }
    
    return paths


def get_extraction_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get PDF extraction configuration."""
    cfg = config or load_config()
    return cfg.get("extraction", {})


def get_chunking_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get text chunking configuration."""
    cfg = config or load_config()
    return cfg.get("chunking", {})


def get_vectorization_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get vectorization/embedding configuration."""
    cfg = config or load_config()
    return cfg.get("vectorization", {})


def get_storage_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get storage configuration."""
    cfg = config or load_config()
    return cfg.get("storage", {})


def get_processing_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get general processing configuration."""
    cfg = config or load_config()
    return cfg.get("processing", {})


# Convenience function to get all paths
def ensure_paths_exist() -> Dict[str, Path]:
    """Ensure all configured directories exist and return paths."""
    paths = get_paths_config()
    
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


if __name__ == "__main__":
    # Test configuration loading
    from pprint import pprint
    
    print("=== Configuration Test ===\n")
    
    config = load_config()
    print("Full config:")
    pprint(config)
    
    print("\n=== Paths ===")
    pprint(get_paths_config())
    
    print("\n=== Extraction Settings ===")
    pprint(get_extraction_config())
    
    print("\n=== Chunking Settings ===")
    pprint(get_chunking_config())
    
    print("\n=== Vectorization Settings ===")
    pprint(get_vectorization_config())
