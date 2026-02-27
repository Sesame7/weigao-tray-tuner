from __future__ import annotations

from typing import Any, Dict, List, Tuple


DEFAULT_SLOTS_PER_ROW = [17, 17, 17]
DEFAULT_SLOT_ROI_WIDTH = 150
DEFAULT_SLOT_ROI_HEIGHT = 200


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _as_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _anchor_xy(anchor: Any) -> Tuple[int, int]:
    if isinstance(anchor, dict):
        return int(anchor.get("x", 0)), int(anchor.get("y", 0))
    if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
        return int(anchor[0]), int(anchor[1])
    return 0, 0


def normalize_slot_layout(params: Dict[str, Any]) -> Dict[str, Any]:
    slot_layout = params.get("slot_layout")
    if not isinstance(slot_layout, dict):
        slot_layout = {}
        params["slot_layout"] = slot_layout

    slots_per_row_raw = slot_layout.get("slots_per_row")
    if not isinstance(slots_per_row_raw, list) or len(slots_per_row_raw) <= 0:
        slots_per_row_raw = list(DEFAULT_SLOTS_PER_ROW)
    slots_per_row = [max(1, _as_int(v, 17)) for v in slots_per_row_raw]
    slot_layout["slots_per_row"] = slots_per_row
    rows = len(slots_per_row)

    slot_roi = slot_layout.get("slot_roi")
    if not isinstance(slot_roi, dict):
        slot_roi = {}
        slot_layout["slot_roi"] = slot_roi
    slot_roi["width_px"] = max(
        1, _as_int(slot_roi.get("width_px"), DEFAULT_SLOT_ROI_WIDTH)
    )
    slot_roi["height_px"] = max(
        1, _as_int(slot_roi.get("height_px"), DEFAULT_SLOT_ROI_HEIGHT)
    )

    def _normalize_anchors(raw: Any) -> List[Dict[str, int]]:
        src = raw if isinstance(raw, list) else []
        out: List[Dict[str, int]] = []
        for i in range(rows):
            item = src[i] if i < len(src) else {}
            x, y = _anchor_xy(item)
            out.append({"x": int(x), "y": int(y)})
        return out

    slot_layout["left_anchors"] = _normalize_anchors(slot_layout.get("left_anchors"))
    slot_layout["right_anchors"] = _normalize_anchors(slot_layout.get("right_anchors"))
    return slot_layout


def clamp_slot_layout_to_image(
    img_w: int, img_h: int, params: Dict[str, Any]
) -> Tuple[int, int]:
    slot_layout = params["slot_layout"]
    roi = slot_layout["slot_roi"]
    w = max(1, min(int(roi["width_px"]), int(img_w)))
    h = max(1, min(int(roi["height_px"]), int(img_h)))
    roi["width_px"] = w
    roi["height_px"] = h

    left_anchors = slot_layout["left_anchors"]
    right_anchors = slot_layout["right_anchors"]
    rows = len(slot_layout["slots_per_row"])
    for r in range(rows):
        left = left_anchors[r]
        right = right_anchors[r]
        xL, yT = int(left["x"]), int(left["y"])
        xR = int(right["x"])
        xL = int(_clamp(xL, 0, max(0, img_w - w)))
        xR = int(_clamp(xR, 0, max(0, img_w - w)))
        yT = int(_clamp(yT, 0, max(0, img_h - h)))
        if xR < xL:
            xR = xL
        left_anchors[r] = {"x": xL, "y": yT}
        right_anchors[r] = {"x": xR, "y": yT}
    return w, h


def generate_grid_rois(
    img_w: int, img_h: int, params: Dict[str, Any]
) -> List[Tuple[int, int, int, int, int, int]]:
    w, h = clamp_slot_layout_to_image(img_w, img_h, params)
    slot_layout = params["slot_layout"]
    cols_per_row = list(slot_layout["slots_per_row"])
    left_anchors = slot_layout["left_anchors"]
    right_anchors = slot_layout["right_anchors"]
    rows = len(cols_per_row)
    rois = []
    for r in range(rows):
        cols = int(cols_per_row[r])
        left = left_anchors[r]
        right = right_anchors[r]
        xL, yT = int(left["x"]), int(left["y"])
        xR = int(right["x"])
        xL = int(_clamp(xL, 0, max(0, img_w - w)))
        xR = int(_clamp(xR, 0, max(0, img_w - w)))
        yT = int(_clamp(yT, 0, max(0, img_h - h)))
        if xR < xL:
            xR = xL
        dx = 0.0 if cols <= 1 else (xR - xL) / (cols - 1)
        for c in range(cols):
            x = int(round(xL + c * dx))
            x = int(_clamp(x, 0, max(0, img_w - w)))
            rois.append((r, c, x, yT, w, h))
    return rois
