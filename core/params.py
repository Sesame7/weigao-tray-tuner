from __future__ import annotations

import copy
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "detect_weigao_tray.yaml"
)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"not a bool: {v!r}")


def _cast_like(default: Any, value: Any) -> Any:
    if value is None:
        return default
    try:
        if isinstance(default, bool):
            return _to_bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
        if isinstance(default, list) and isinstance(value, list):
            if len(default) == 0:
                return copy.deepcopy(value)
            if len(default) != len(value):
                return [_cast_like(default[0], v) for v in value]
            return [_cast_like(d, v) for d, v in zip(default, value)]
    except Exception:
        return default
    return copy.deepcopy(value)


def _merge_with_defaults(
    default: Dict[str, Any], data: Dict[str, Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(default)
    for k, v in data.items():
        if k in default:
            dv = default[k]
            if isinstance(dv, dict) and isinstance(v, dict):
                out[k] = _merge_with_defaults(dv, v)
            else:
                out[k] = _cast_like(dv, v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml_dict(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        traceback.print_exc()
        return {}


def load_effective_params(
    path: str, defaults_path: str | None = None
) -> Dict[str, Any]:
    template_path = Path(defaults_path) if defaults_path else DEFAULT_TEMPLATE_PATH
    defaults = _read_yaml_dict(template_path) if template_path.exists() else {}

    cfg_path = Path(path)
    data = _read_yaml_dict(cfg_path) if cfg_path.exists() else {}

    if defaults:
        return _merge_with_defaults(defaults, data)
    if data:
        return copy.deepcopy(data)
    return {}


def save_params(path: str, params: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(target) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    tmp.replace(target)
