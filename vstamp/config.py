from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, recursively resolving an optional ``base`` config."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    base_ref = cfg.pop("base", None)
    if base_ref:
        base_path = (cfg_path.parent / base_ref).resolve()
        cfg = _merge(load_config(base_path), cfg)
    cfg["_config_path"] = str(cfg_path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    required = ("dataset", "data", "model", "training", "matching", "loss", "evaluation")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    resolutions = list(cfg["data"].get("resolutions_ms", []))
    if resolutions != [100, 500, 2000]:
        raise ValueError("Paper configuration requires resolutions_ms=[100, 500, 2000]")
    if int(cfg["data"].get("observation_seconds", 0)) != 60:
        raise ValueError("Paper configuration requires a 60 second observation")
    if int(cfg["model"].get("tokens", 0)) <= 0:
        raise ValueError("model.tokens must be positive")


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(clean, stream, sort_keys=False)

