from __future__ import annotations

from typing import Any, Dict

import numpy as np

from detect.base import Detector
from detect.weigao_tray import WeigaoTrayDetector


def create_runtime_detector(params: Dict[str, Any]) -> Detector:
    return WeigaoTrayDetector(
        params=params,
        generate_overlay=True,
        input_pixel_format="bgr8",
    )


def run_detector(detector_instance: Detector, img_bgr: np.ndarray) -> Dict[str, Any]:
    ok, message, overlay_bgr, result_code = detector_instance.detect(img_bgr)
    if overlay_bgr is None:
        overlay_bgr = img_bgr.copy()
    return {
        "ok": bool(ok),
        "message": str(message),
        "result_code": str(result_code or ("OK" if ok else "NG")),
        "overlay_bgr": overlay_bgr,
    }


def inspect_image(img_bgr: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    return run_detector(create_runtime_detector(params), img_bgr)
