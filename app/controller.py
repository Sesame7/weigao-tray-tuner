from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

from core import detector
from core import params as params_mod
from core.slot_layout_utils import (
    clamp_slot_layout_to_image,
    generate_grid_rois,
)
from app.image_session import ImageSession
from ui.control_panel import ControlPanel
from ui.image_view import ImageView


def _source_base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_base_dir() -> Path:
    # PyInstaller: use the executable directory for user-editable files.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return _source_base_dir()


BASE_DIR = _runtime_base_dir()
SYNC_CONFIG_NAME = "detect_weigao_tray.yaml"
SYNC_CONFIG_PATH = BASE_DIR / SYNC_CONFIG_NAME
if not SYNC_CONFIG_PATH.exists():
    for candidate in (BASE_DIR / "config" / SYNC_CONFIG_NAME,):
        if candidate.exists():
            SYNC_CONFIG_PATH = candidate
            break
WORK_CONFIG_PATH = BASE_DIR / "config.yaml"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = str(DATA_DIR / "images")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DBG_COLOR = (0, 0, 200)  # RGB
GUIDANCE_ADJUST = "Adjust anchors or params, then click Detect (Ctrl+R)."
GUIDANCE_REVIEW = "Review the overlay, adjust if needed, then Detect again."


class DisplayState:
    def __init__(self) -> None:
        self.source_bgr: Optional[np.ndarray] = None
        self.showing_source_image = False
        self.scale_x = 1.0
        self.scale_y = 1.0

    def set_source(self, bgr: np.ndarray) -> None:
        self.source_bgr = bgr
        self.showing_source_image = False
        self.scale_x = 1.0
        self.scale_y = 1.0

    def set_display_image(self, img_bgr: Optional[np.ndarray]) -> None:
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.showing_source_image = (
            self.source_bgr is not None
            and img_bgr is not None
            and img_bgr is self.source_bgr
        )
        if self.source_bgr is None or img_bgr is None:
            return
        src_h, src_w = self.source_bgr.shape[:2]
        dst_h, dst_w = img_bgr.shape[:2]
        if src_w > 0 and src_h > 0:
            self.scale_x = float(dst_w) / float(src_w)
            self.scale_y = float(dst_h) / float(src_h)

    def to_display_point(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(round(float(x) * self.scale_x)),
            int(round(float(y) * self.scale_y)),
        )

    def to_source_point(self, x: float, y: float) -> tuple[float, float]:
        sx = self.scale_x if self.scale_x > 0 else 1.0
        sy = self.scale_y if self.scale_y > 0 else 1.0
        return float(x) / sx, float(y) / sy


class TuningState:
    def __init__(
        self,
        *,
        load_path: Path,
        save_path: Path,
        defaults_path: Path,
    ) -> None:
        self.load_path = Path(load_path)
        self.save_path = Path(save_path)
        self.defaults_path = Path(defaults_path)
        self.params: Dict[str, Any] = params_mod.load_effective_params(
            str(self.load_path),
            defaults_path=str(self.defaults_path),
        )
        self._dirty = False

    def mark_dirty(self) -> None:
        self._dirty = True

    def save_if_dirty(self) -> bool:
        if not self._dirty:
            return False
        params_mod.save_params(str(self.save_path), self.params)
        self._dirty = False
        return True


def is_text_input_focused(root: tk.Misc) -> bool:
    widget = root.focus_get()
    return isinstance(widget, (tk.Entry, ttk.Entry))


def bind_app_shortcuts(
    root: tk.Misc,
    *,
    on_open,
    on_run,
    on_prev,
    on_next,
    should_ignore_lr,
) -> None:
    root.bind("<Control-o>", lambda _e: on_open())
    root.bind("<Control-r>", lambda _e: on_run())

    def _on_prev(_event) -> None:
        if should_ignore_lr():
            return
        on_prev()

    def _on_next(_event) -> None:
        if should_ignore_lr():
            return
        on_next()

    root.bind("<Left>", _on_prev)
    root.bind("<Right>", _on_next)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Weigao Tray Tuner")
        self.config_load_path = (
            WORK_CONFIG_PATH if WORK_CONFIG_PATH.exists() else SYNC_CONFIG_PATH
        )
        self.config_save_path = WORK_CONFIG_PATH
        self.tuning = TuningState(
            load_path=self.config_load_path,
            save_path=self.config_save_path,
            defaults_path=SYNC_CONFIG_PATH,
        )
        self.display = DisplayState()
        self.images = ImageSession(suffixes=IMAGE_SUFFIXES)
        self._detector_instance = None
        self._detector_dirty = True

        self.main = ttk.Frame(root, padding=6)
        self.main.pack(fill="both", expand=True)
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=0)
        self.main.rowconfigure(0, weight=1)

        self.view = ImageView(
            self.main, self._on_anchor_drag, self.prev_image, self.next_image
        )
        self.view.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.panel = ControlPanel(
            self.main,
            self.params,
            self._mark_dirty,
            self.open_image,
            self.run,
        )
        self.panel.frame.grid(row=0, column=1, sticky="nsew")
        self.panel.set_detect_enabled(False)
        self._bind_shortcuts()

        self._refresh_anchors()
        self._refresh_grid_overlays()

    @property
    def params(self) -> Dict[str, Any]:
        return self.tuning.params

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return lo if v < lo else hi if v > hi else v

    def _handle_error(self, *, redraw: bool = False) -> None:
        traceback.print_exc()
        if redraw:
            self.view.redraw()

    def _set_adjust_guidance(self) -> None:
        self.panel.set_guidance(GUIDANCE_ADJUST)

    def _get_clamped_roi_size(self, img_w: int, img_h: int) -> tuple[int, int]:
        return clamp_slot_layout_to_image(img_w, img_h, self.params)

    def _clamp_anchor_xy(
        self, ix: float, iy: float, img_w: int, img_h: int, roi_w: int, roi_h: int
    ) -> tuple[int, int]:
        x = self._clamp(int(round(ix)), 0, max(0, img_w - roi_w))
        y = self._clamp(int(round(iy)), 0, max(0, img_h - roi_h))
        return x, y

    def _refresh_editor_view(
        self, *, show_source_image: bool, update_nav: bool
    ) -> None:
        if (
            show_source_image
            and self.display.source_bgr is not None
            and not self.display.showing_source_image
        ):
            self._set_display_image(self.display.source_bgr)
        self._refresh_anchors()
        self._refresh_grid_overlays()
        if update_nav:
            self._update_nav_buttons()

    def _set_display_image(self, img_bgr: Optional[np.ndarray]) -> None:
        self.view.set_image(img_bgr)
        self.display.set_display_image(img_bgr)

    def _bind_shortcuts(self) -> None:
        bind_app_shortcuts(
            self.root,
            on_open=self.open_image,
            on_run=self.run,
            on_prev=self.prev_image,
            on_next=self.next_image,
            should_ignore_lr=lambda: is_text_input_focused(self.root),
        )

    def _update_nav_buttons(self) -> None:
        self.view.set_nav_enabled(self.images.has_prev, self.images.has_next)

    def _after_image_loaded(self, *, update_nav: bool) -> None:
        self._set_adjust_guidance()
        self.panel.set_detect_enabled(True)
        self._refresh_editor_view(show_source_image=True, update_nav=update_nav)

    def _go_to_image(self, index: int) -> None:
        try:
            moved = self._go_to_index(index)
            if not moved:
                return
            self._after_image_loaded(update_nav=True)
        except Exception:
            self._update_nav_buttons()
            self._handle_error()

    def _load_selected_image(self, path: Path) -> None:
        try:
            self._load_image_path(path, rebuild_list=True)
            self._after_image_loaded(update_nav=True)
        except Exception:
            self._handle_error()

    def prev_image(self) -> None:
        self._go_to_image(self.images.index - 1)

    def next_image(self) -> None:
        self._go_to_image(self.images.index + 1)

    def _mark_dirty(self) -> None:
        self.tuning.mark_dirty()
        self._detector_dirty = True
        self._set_adjust_guidance()
        self._refresh_editor_view(show_source_image=True, update_nav=False)

    def _refresh_anchors(self) -> None:
        if self.display.source_bgr is None:
            self.view.set_anchors([])
            return
        h, w = self.display.source_bgr.shape[:2]
        self._get_clamped_roi_size(w, h)
        anchors = []
        slot_layout = self.params["slot_layout"]
        left_anchors = slot_layout["left_anchors"]
        right_anchors = slot_layout["right_anchors"]
        rows = len(slot_layout["slots_per_row"])
        for r in range(rows):
            left = left_anchors[r]
            right = right_anchors[r]
            xL, yT = int(left["x"]), int(left["y"])
            xR, _ = int(right["x"]), int(right["y"])
            xL_v, yT_v = self.display.to_display_point(xL, yT)
            xR_v, _ = self.display.to_display_point(xR, yT)
            anchors.append((r, "L", xL_v, yT_v))
            anchors.append((r, "R", xR_v, yT_v))
        self.view.set_anchors(anchors)

    def _refresh_grid_overlays(self) -> None:
        if self.display.source_bgr is None:
            self.view.set_overlays([])
            return
        show_roi = bool(self.params.get("debug_overlay", {}).get("slot_layout", False))
        if not show_roi:
            self.view.set_overlays([])
            return
        h, w = self.display.source_bgr.shape[:2]
        rois = generate_grid_rois(w, h, self.params)
        ovs: List[Dict[str, Any]] = []
        for _r, _c, x, y, rw, rh in rois:
            x_v, y_v = self.display.to_display_point(x, y)
            w_v, h_v = self.display.to_display_point(rw, rh)
            ovs.append(
                {
                    "type": "rect",
                    "x": x_v,
                    "y": y_v,
                    "w": max(1, w_v),
                    "h": max(1, h_v),
                    "color": DBG_COLOR,
                    "width": 1,
                }
            )
        self.view.set_overlays(ovs)

    def _on_anchor_drag(self, row: int, side: str, ix: float, iy: float) -> None:
        if self.display.source_bgr is None:
            return
        h, w = self.display.source_bgr.shape[:2]
        rw, rh = self._get_clamped_roi_size(w, h)
        src_x, src_y = self.display.to_source_point(ix, iy)
        x, y = self._clamp_anchor_xy(src_x, src_y, w, h, rw, rh)
        slot_layout = self.params["slot_layout"]
        left_anchors = slot_layout["left_anchors"]
        right_anchors = slot_layout["right_anchors"]
        xL, _ = int(left_anchors[row]["x"]), int(left_anchors[row]["y"])
        xR, _ = int(right_anchors[row]["x"]), int(right_anchors[row]["y"])
        if side == "L":
            xL = x
            if xR < xL:
                xR = xL
        else:
            xR = x
            if xR < xL:
                xL = xR
        left_anchors[row] = {"x": xL, "y": y}
        right_anchors[row] = {"x": xR, "y": y}
        self._mark_dirty()

    def _get_initial_open_dir(self) -> str:
        current = self.images.current_path
        if current is not None:
            return str(current.parent)
        if Path(SAMPLES_DIR).exists():
            return SAMPLES_DIR
        if DATA_DIR.exists():
            return str(DATA_DIR)
        return str(BASE_DIR)

    def open_image(self) -> None:
        initial_dir = self._get_initial_open_dir()
        path = self._open_image_dialog(initial_dir=initial_dir)
        if path is None:
            return
        self._load_selected_image(path)

    def run(self) -> None:
        if self.display.source_bgr is None:
            return
        try:
            overlay_bgr = self._run_detection()
            self._set_display_image(overlay_bgr)
            self._refresh_anchors()
            self.panel.set_guidance(GUIDANCE_REVIEW)
            self.tuning.save_if_dirty()
            self._refresh_grid_overlays()
        except Exception:
            self._handle_error(redraw=True)

    def _open_image_dialog(self, *, initial_dir: str) -> Path | None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
            initialdir=initial_dir,
        )
        if not path:
            return None
        return Path(path)

    def _load_image_path(self, path: Path, *, rebuild_list: bool) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        if rebuild_list:
            self.images.build_around(path)
        self.display.set_source(bgr)

    def _go_to_index(self, index: int) -> bool:
        if not self.images.can_index(index):
            return False
        prev_index = self.images.index
        self.images.set_index(index)
        current_path = self.images.current_path
        if current_path is None:
            self.images.restore_index(prev_index)
            return False
        try:
            self._load_image_path(current_path, rebuild_list=False)
        except Exception:
            self.images.restore_index(prev_index)
            raise
        return True

    def _run_detection(self) -> np.ndarray:
        if self.display.source_bgr is None:
            raise RuntimeError("No source image loaded.")
        if self._detector_instance is None or self._detector_dirty:
            self._detector_instance = detector.create_runtime_detector(self.params)
            self._detector_dirty = False
        result = detector.run_detector(self._detector_instance, self.display.source_bgr)
        overlay_bgr = result["overlay_bgr"]
        return overlay_bgr
