"""Config loading. Everything resolves relative to the repo, never to a machine path."""
import copy
from pathlib import Path

import yaml

HW_ROOT = Path(__file__).resolve().parents[1]          # CMS/hardware
CMS_ROOT = HW_ROOT.parent                              # CMS
SCRIPTS_DIR = CMS_ROOT / "scripts"
FLIGHTS_DIR = HW_ROOT / "flights"
DEFAULT_CONFIG = HW_ROOT / "config.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None) -> dict:
    """Load config.yaml (or `path`), then apply a nested `overrides` dict (used by tests/CLI)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _merge(cfg, overrides or {})


def save_config_values(updates: dict, path=None):
    """Merge `updates` (nested dict) into config.yaml. Used by the calibration scripts.

    Rewrites the file with yaml.safe_dump, so comments are lost; calibration scripts
    therefore write to config.local.yaml instead, which load_config layers on top.
    """
    path = Path(path) if path else HW_ROOT / "config.local.yaml"
    cur = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cur = yaml.safe_load(f) or {}
    cur = _merge(cur, updates)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cur, f, sort_keys=False)
    return path


def load_effective_config(overrides=None) -> dict:
    """config.yaml, then config.local.yaml (calibration results), then overrides."""
    cfg = load_config()
    local = HW_ROOT / "config.local.yaml"
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _merge(cfg, overrides or {})
