from __future__ import annotations

from typing import Any, Dict, Mapping

from ui.param_schema import FieldSpec, row_count_from_params


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


class ParamState:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def row_count(self) -> int:
        return row_count_from_params(self.params)

    def get_default(self, path: str, default: Any) -> Any:
        try:
            return self.get(path)
        except Exception:
            return default

    def get_bool(self, path: str) -> bool:
        return bool(self.get_default(path, False))

    def set_bool(self, path: str, value: bool) -> None:
        self.set(path, bool(value))

    def get(self, path: str) -> Any:
        cur: Any = self.params
        for part in path.split("."):
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        return cur

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        cur: Any = self.params
        for part in parts[:-1]:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        last = parts[-1]
        if isinstance(cur, list):
            cur[int(last)] = value
        else:
            cur[last] = value

    def display_text(self, field: FieldSpec) -> str:
        kind = field.kind
        paths = field.paths
        if kind == "int":
            return str(int(self.get_default(paths[0], 0)))
        if kind == "float":
            return str(float(self.get_default(paths[0], 0.0)))
        if kind == "csv2":
            return (
                f"{int(self.get_default(paths[0], 0))},"
                f"{int(self.get_default(paths[1], 0))}"
            )
        if kind == "csv2f":
            return (
                f"{float(self.get_default(paths[0], 0.0))},"
                f"{float(self.get_default(paths[1], 0.0))}"
            )
        if kind == "csv3":
            values = self.get_default(paths[0], [0, 0, 0])
            if not isinstance(values, list):
                values = [0, 0, 0]
            return ",".join(str(int(v)) for v in values)
        if kind == "csvN":
            values = self.get_default(paths[0], [17, 17, 17])
            if not isinstance(values, list):
                values = [17, 17, 17]
            return ",".join(str(int(v)) for v in values)
        raise ValueError(f"display_text not supported for kind={kind}")

    def update_from_text(self, field: FieldSpec, text: str) -> bool:
        kind = field.kind
        paths = field.paths
        if kind == "int":
            self.set(paths[0], int(float(text)))
            return False
        if kind == "float":
            self.set(paths[0], float(text))
            return False
        if kind == "csv2":
            parts = [p.strip() for p in text.split(",")]
            if len(parts) != 2:
                raise ValueError("csv2 expects 2 values")
            self.set(paths[0], int(float(parts[0])))
            self.set(paths[1], int(float(parts[1])))
            return False
        if kind == "csv2f":
            parts = [p.strip() for p in text.split(",")]
            if len(parts) != 2:
                raise ValueError("csv2f expects 2 values")
            self.set(paths[0], float(parts[0]))
            self.set(paths[1], float(parts[1]))
            return False
        if kind == "csv3":
            parts = [p.strip() for p in text.split(",")]
            if len(parts) != 3:
                raise ValueError("csv3 expects 3 values")
            self.set(paths[0], [int(float(p)) for p in parts])
            return False
        if kind == "csvN":
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) <= 0:
                raise ValueError("csvN expects at least 1 value")
            old_row_count = self.row_count()
            values = [int(float(p)) for p in parts]
            self.set(paths[0], values)
            if paths[0] == "slot_layout.slots_per_row":
                self._sync_junction_white_by_row(len(values))
                self._sync_slot_anchors_by_row(len(values))
            return self.row_count() != old_row_count
        raise ValueError(f"Unsupported field kind: {kind}")

    def _sync_junction_white_by_row(self, rows: int) -> None:
        junction_line = self.params.get("junction_line")
        if not isinstance(junction_line, dict):
            junction_line = {}
            self.params["junction_line"] = junction_line

        white_by_row = junction_line.get("white_by_row")
        old_rows = white_by_row if isinstance(white_by_row, list) else []
        new_rows: list[dict[str, int]] = []
        for i in range(rows):
            src = old_rows[i] if i < len(old_rows) else {}
            if not isinstance(src, Mapping):
                src = {}
            new_rows.append(
                {
                    "s_max": _coerce_int(src.get("s_max"), 50),
                    "v_min": _coerce_int(src.get("v_min"), 50),
                }
            )
        junction_line["white_by_row"] = new_rows

    def _sync_slot_anchors_by_row(self, rows: int) -> None:
        slot_layout = self.params.get("slot_layout")
        if not isinstance(slot_layout, dict):
            slot_layout = {}
            self.params["slot_layout"] = slot_layout

        def _sync_one_side(key: str) -> None:
            raw = slot_layout.get(key)
            src = raw if isinstance(raw, list) else []
            out: list[dict[str, int]] = []
            for i in range(rows):
                item = src[i] if i < len(src) else (src[-1] if src else {})
                if not isinstance(item, Mapping):
                    item = {}
                out.append(
                    {
                        "x": _coerce_int(item.get("x"), 0),
                        "y": _coerce_int(item.get("y"), 0),
                    }
                )
            slot_layout[key] = out

        _sync_one_side("left_anchors")
        _sync_one_side("right_anchors")
