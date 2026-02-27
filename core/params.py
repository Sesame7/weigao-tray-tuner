from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "detect_weigao_tray.yaml"
)


class _CompactDumper(yaml.SafeDumper):
    pass


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _is_scalar_list(v: Any, max_len: int = 8) -> bool:
    return isinstance(v, list) and len(v) <= max_len and all(_is_scalar(i) for i in v)


def _is_simple_mapping(v: Any, max_len: int = 3) -> bool:
    if not isinstance(v, dict) or len(v) > max_len:
        return False
    for k, val in v.items():
        if not _is_scalar(k):
            return False
        if _is_scalar(val):
            continue
        if _is_scalar_list(val):
            continue
        return False
    return True


def _represent_compact_list(dumper: _CompactDumper, data: list) -> yaml.Node:
    flow = all(_is_scalar(i) for i in data) and len(data) <= 8
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        data,
        flow_style=flow,
    )


def _represent_compact_dict(dumper: _CompactDumper, data: dict) -> yaml.Node:
    flow = _is_simple_mapping(data)
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        data.items(),
        flow_style=flow,
    )


_CompactDumper.add_representer(list, _represent_compact_list)
_CompactDumper.add_representer(dict, _represent_compact_dict)


def _read_yaml_dict_strict(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be mapping: {path}")
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_with_defaults_strict(default: Any, data: Any, *, path: str) -> Any:
    if isinstance(default, dict):
        if not isinstance(data, dict):
            raise TypeError(f"Expected mapping at '{path}', got {type(data).__name__}.")
        out: Dict[str, Any] = copy.deepcopy(default)
        for key, value in data.items():
            if key not in default:
                raise KeyError(f"Unknown key '{path}.{key}'.")
            out[key] = _merge_with_defaults_strict(
                default[key], value, path=f"{path}.{key}"
            )
        return out

    if isinstance(default, list):
        if not isinstance(data, list):
            raise TypeError(f"Expected list at '{path}', got {type(data).__name__}.")
        if len(default) == 0:
            return copy.deepcopy(data)
        item_template = default[0]
        return [
            _merge_with_defaults_strict(item_template, item, path=f"{path}[{i}]")
            for i, item in enumerate(data)
        ]

    if isinstance(default, bool):
        if not isinstance(data, bool):
            raise TypeError(f"Expected bool at '{path}', got {type(data).__name__}.")
        return data

    if _is_number(default):
        if not _is_number(data):
            raise TypeError(
                f"Expected numeric value at '{path}', got {type(data).__name__}."
            )
        return data

    if isinstance(default, str):
        if not isinstance(data, str):
            raise TypeError(f"Expected string at '{path}', got {type(data).__name__}.")
        return data

    if default is None:
        return copy.deepcopy(data)

    if not isinstance(data, type(default)):
        raise TypeError(
            f"Expected {type(default).__name__} at '{path}', got {type(data).__name__}."
        )
    return copy.deepcopy(data)


def _require_mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping at '{path}', got {type(value).__name__}.")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"Expected list at '{path}', got {type(value).__name__}.")
    return value


def _require_int(value: Any, path: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected int at '{path}', got {type(value).__name__}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"Expected '{path}' >= {min_value}, got {value}.")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool at '{path}', got {type(value).__name__}.")
    return value


def _validate_runtime_params(params: Dict[str, Any]) -> None:
    slot_layout = _require_mapping(params.get("slot_layout"), "slot_layout")
    slots_per_row = _require_list(
        slot_layout.get("slots_per_row"), "slot_layout.slots_per_row"
    )
    if len(slots_per_row) <= 0:
        raise ValueError("'slot_layout.slots_per_row' cannot be empty.")
    for i, value in enumerate(slots_per_row):
        _require_int(value, f"slot_layout.slots_per_row[{i}]", min_value=1)

    rows = len(slots_per_row)
    slot_roi = _require_mapping(slot_layout.get("slot_roi"), "slot_layout.slot_roi")
    _require_int(slot_roi.get("width_px"), "slot_layout.slot_roi.width_px", min_value=1)
    _require_int(
        slot_roi.get("height_px"), "slot_layout.slot_roi.height_px", min_value=1
    )

    left_anchors = _require_list(
        slot_layout.get("left_anchors"), "slot_layout.left_anchors"
    )
    right_anchors = _require_list(
        slot_layout.get("right_anchors"), "slot_layout.right_anchors"
    )
    if len(left_anchors) != rows:
        raise ValueError(
            f"'slot_layout.left_anchors' count must match rows ({rows}), got {len(left_anchors)}."
        )
    if len(right_anchors) != rows:
        raise ValueError(
            f"'slot_layout.right_anchors' count must match rows ({rows}), got {len(right_anchors)}."
        )
    for i, anchor in enumerate(left_anchors):
        a = _require_mapping(anchor, f"slot_layout.left_anchors[{i}]")
        _require_int(a.get("x"), f"slot_layout.left_anchors[{i}].x")
        _require_int(a.get("y"), f"slot_layout.left_anchors[{i}].y")
    for i, anchor in enumerate(right_anchors):
        a = _require_mapping(anchor, f"slot_layout.right_anchors[{i}]")
        _require_int(a.get("x"), f"slot_layout.right_anchors[{i}].x")
        _require_int(a.get("y"), f"slot_layout.right_anchors[{i}].y")

    junction_line = _require_mapping(params.get("junction_line"), "junction_line")
    white_by_row = _require_list(
        junction_line.get("white_by_row"), "junction_line.white_by_row"
    )
    if len(white_by_row) != rows:
        raise ValueError(
            f"'junction_line.white_by_row' count must match rows ({rows}), got {len(white_by_row)}."
        )
    for i, row_item in enumerate(white_by_row):
        item = _require_mapping(row_item, f"junction_line.white_by_row[{i}]")
        _require_int(item.get("s_max"), f"junction_line.white_by_row[{i}].s_max")
        _require_int(item.get("v_min"), f"junction_line.white_by_row[{i}].v_min")

    debug_overlay = _require_mapping(params.get("debug_overlay"), "debug_overlay")
    for key in ("slot_layout", "stem_mask", "junction_line", "color_band"):
        _require_bool(debug_overlay.get(key), f"debug_overlay.{key}")


def load_effective_params(
    path: str, defaults_path: str | None = None
) -> Dict[str, Any]:
    template_path = Path(defaults_path) if defaults_path else DEFAULT_TEMPLATE_PATH
    if not template_path.exists():
        raise FileNotFoundError(f"Defaults config not found: {template_path}")
    defaults = _read_yaml_dict_strict(template_path)

    cfg_path = Path(path)
    data = _read_yaml_dict_strict(cfg_path) if cfg_path.exists() else {}

    params = _merge_with_defaults_strict(defaults, data, path="root")
    _validate_runtime_params(params)
    return params


def save_params(path: str, params: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(target) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(
            params,
            f,
            Dumper=_CompactDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=120,
        )
    tmp.replace(target)
