"""Configuration loading: config.yaml + .env + PA__ env overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _coerce(val: str) -> Any:
    return yaml.safe_load(val)


def _apply_env_overrides(cfg: dict) -> None:
    for key, val in os.environ.items():
        if not key.startswith("PA__"):
            continue
        parts = key[4:].split("__")
        node = cfg
        for i, p in enumerate(parts):
            # Match existing keys case-insensitively (e.g. srlm.K); new keys are lower-case.
            match = next((k for k in node if k.lower() == p.lower()), p.lower())
            if i == len(parts) - 1:
                node[match] = _coerce(val)
            else:
                node = node.setdefault(match, {})


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: base recursively updated by override."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config, apply env overrides and explicit overrides."""
    load_dotenv()
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _apply_env_overrides(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def secret(name: str) -> str | None:
    """Read a secret from the environment (.env loaded by load_config)."""
    load_dotenv()
    v = os.environ.get(name)
    return v or None
