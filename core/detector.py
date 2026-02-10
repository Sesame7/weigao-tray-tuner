from __future__ import annotations

from typing import Any, Dict

import numpy as np

from detect.weigao_tray import WeigaoTrayDetector


def inspect_image(img_bgr: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    detector = WeigaoTrayDetector(
        params=params,
        generate_overlay=True,
        input_pixel_format="bgr8",
    )
    ok, message, overlay_bgr, result_code = detector.detect(img_bgr)
    if overlay_bgr is None:
        overlay_bgr = img_bgr.copy()
    return {
        "ok": bool(ok),
        "message": str(message),
        "result_code": str(result_code or ("OK" if ok else "NG")),
        "overlay_bgr": overlay_bgr,
    }
