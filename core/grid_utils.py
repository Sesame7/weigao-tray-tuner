from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def generate_grid_rois(
    img_w: int, img_h: int, params: Dict[str, Any]
) -> List[Tuple[int, int, int, int, int, int]]:
    g = params["grid"]
    rows = int(g["rows"])
    cols_per_row = g["cols_per_row"]
    w = max(1, min(int(g["roi"]["w"]), int(img_w)))
    h = max(1, min(int(g["roi"]["h"]), int(img_h)))
    rois = []
    for r in range(rows):
        cols = int(cols_per_row[r])
        xL, yT = g["anchor_left"][r]
        xR, _yT2 = g["anchor_right"][r]
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
