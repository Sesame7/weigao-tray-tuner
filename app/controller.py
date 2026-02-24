from __future__ import annotations

import re
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
from core.grid_utils import generate_grid_rois
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


def _natural_key(name: str) -> List[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Weigao Tray Tuner")
        self.config_load_path = (
            WORK_CONFIG_PATH if WORK_CONFIG_PATH.exists() else SYNC_CONFIG_PATH
        )
        self.config_save_path = WORK_CONFIG_PATH
        self.params: Dict[str, Any] = params_mod.load_effective_params(
            str(self.config_load_path), defaults_path=str(SYNC_CONFIG_PATH)
        )
        self.dirty = True
        self.params_dirty = False
        self.img_rgb: Optional[np.ndarray] = None
        self.img_bgr: Optional[np.ndarray] = None
        self.image_paths: List[Path] = []
        self.image_index: int = -1

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
        rw = self._clamp(int(self.params["grid"]["roi"]["w"]), 1, img_w)
        rh = self._clamp(int(self.params["grid"]["roi"]["h"]), 1, img_h)
        self.params["grid"]["roi"]["w"] = rw
        self.params["grid"]["roi"]["h"] = rh
        return rw, rh

    def _clamp_anchor_xy(
        self, ix: float, iy: float, img_w: int, img_h: int, roi_w: int, roi_h: int
    ) -> tuple[int, int]:
        x = self._clamp(int(round(ix)), 0, max(0, img_w - roi_w))
        y = self._clamp(int(round(iy)), 0, max(0, img_h - roi_h))
        return x, y

    def _refresh_editor_view(
        self, *, show_source_image: bool, update_nav: bool
    ) -> None:
        if show_source_image and self.img_rgb is not None:
            self.view.set_image(self.img_rgb)
        self._refresh_anchors()
        self._refresh_grid_overlays()
        if update_nav:
            self._update_nav_buttons()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<Control-r>", lambda _e: self.run())
        self.root.bind("<Left>", self._on_prev_shortcut)
        self.root.bind("<Right>", self._on_next_shortcut)

    def _focus_is_text_input(self) -> bool:
        w = self.root.focus_get()
        return isinstance(w, (tk.Entry, ttk.Entry))

    def _on_prev_shortcut(self, _event) -> None:
        if self._focus_is_text_input():
            return
        self.prev_image()

    def _on_next_shortcut(self, _event) -> None:
        if self._focus_is_text_input():
            return
        self.next_image()

    def _build_image_list(self, selected_path: Path) -> None:
        folder = selected_path.parent
        candidates = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        candidates.sort(key=lambda p: _natural_key(p.name))
        self.image_paths = candidates
        self.image_index = -1
        selected_abs = str(selected_path.resolve())
        for i, p in enumerate(self.image_paths):
            if str(p.resolve()) == selected_abs:
                self.image_index = i
                break
        if self.image_index < 0:
            self.image_paths.append(selected_path)
            self.image_paths.sort(key=lambda p: _natural_key(p.name))
            self.image_index = next(
                i
                for i, p in enumerate(self.image_paths)
                if str(p.resolve()) == selected_abs
            )

    def _update_nav_buttons(self) -> None:
        total = len(self.image_paths)
        has_prev = total > 1 and self.image_index > 0
        has_next = total > 1 and self.image_index >= 0 and self.image_index < total - 1
        self.view.set_nav_enabled(has_prev, has_next)

    def _load_image_path(self, path: Path, rebuild_list: bool) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if rebuild_list:
            self._build_image_list(path)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        self.img_bgr = bgr
        self.img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.dirty = True
        self._set_adjust_guidance()
        self.panel.set_detect_enabled(True)
        self._refresh_editor_view(show_source_image=True, update_nav=True)

    def _go_to_image(self, index: int) -> None:
        if index < 0 or index >= len(self.image_paths):
            return
        self.image_index = index
        try:
            self._load_image_path(
                self.image_paths[self.image_index], rebuild_list=False
            )
        except Exception:
            self._handle_error()

    def prev_image(self) -> None:
        self._go_to_image(self.image_index - 1)

    def next_image(self) -> None:
        self._go_to_image(self.image_index + 1)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.params_dirty = True
        self._set_adjust_guidance()
        self._refresh_editor_view(show_source_image=True, update_nav=False)

    def _refresh_anchors(self) -> None:
        if self.img_bgr is None:
            self.view.set_anchors([])
            return
        h, w = self.img_bgr.shape[:2]
        rw, rh = self._get_clamped_roi_size(w, h)
        anchors = []
        for r in range(int(self.params["grid"]["rows"])):
            xL, yT = self.params["grid"]["anchor_left"][r]
            xR, _ = self.params["grid"]["anchor_right"][r]
            xL = self._clamp(int(xL), 0, max(0, w - rw))
            xR = self._clamp(int(xR), 0, max(0, w - rw))
            yT = self._clamp(int(yT), 0, max(0, h - rh))
            if xR < xL:
                xR = xL
            self.params["grid"]["anchor_left"][r] = [xL, yT]
            self.params["grid"]["anchor_right"][r] = [xR, yT]
            anchors.append((r, "L", xL, yT))
            anchors.append((r, "R", xR, yT))
        self.view.set_anchors(anchors)

    def _refresh_grid_overlays(self) -> None:
        if self.img_bgr is None:
            self.view.set_overlays([])
            return
        h, w = self.img_bgr.shape[:2]
        rois = generate_grid_rois(w, h, self.params)
        show_roi = bool(self.params.get("dbg", {}).get("show_roi", False))
        ovs: List[Dict[str, Any]] = (
            [
                {
                    "type": "rect",
                    "x": x,
                    "y": y,
                    "w": rw,
                    "h": rh,
                    "color": DBG_COLOR,
                    "width": 1,
                }
                for _r, _c, x, y, rw, rh in rois
            ]
            if show_roi
            else []
        )
        self.view.set_overlays(ovs)

    def _on_anchor_drag(self, row: int, side: str, ix: float, iy: float) -> None:
        if self.img_bgr is None:
            return
        h, w = self.img_bgr.shape[:2]
        rw, rh = self._get_clamped_roi_size(w, h)
        x, y = self._clamp_anchor_xy(ix, iy, w, h, rw, rh)
        xL, _ = self.params["grid"]["anchor_left"][row]
        xR, _ = self.params["grid"]["anchor_right"][row]
        if side == "L":
            xL = x
            if xR < xL:
                xR = xL
        else:
            xR = x
            if xR < xL:
                xL = xR
        self.params["grid"]["anchor_left"][row] = [xL, y]
        self.params["grid"]["anchor_right"][row] = [xR, y]
        self._mark_dirty()

    def _get_initial_open_dir(self) -> str:
        if 0 <= self.image_index < len(self.image_paths):
            return str(self.image_paths[self.image_index].parent)
        if Path(SAMPLES_DIR).exists():
            return SAMPLES_DIR
        if DATA_DIR.exists():
            return str(DATA_DIR)
        return str(BASE_DIR)

    def open_image(self) -> None:
        initial_dir = self._get_initial_open_dir()
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
            initialdir=initial_dir,
        )
        if not path:
            return
        try:
            self._load_image_path(Path(path), rebuild_list=True)
        except Exception:
            self._handle_error()

    def run(self) -> None:
        if self.img_bgr is None:
            return
        try:
            result = detector.inspect_image(self.img_bgr, self.params)
            overlay_bgr = result["overlay_bgr"]
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            self.view.set_image(overlay_rgb)
            self.dirty = False
            self.panel.set_guidance(GUIDANCE_REVIEW)
            if self.params_dirty:
                params_mod.save_params(str(self.config_save_path), self.params)
                self.params_dirty = False
            self._refresh_grid_overlays()
        except Exception:
            self._handle_error(redraw=True)
