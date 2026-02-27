from __future__ import annotations

import base64
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk


class ImageView(tk.Canvas):
    def __init__(self, master, on_anchor_drag, on_prev_click=None, on_next_click=None):
        super().__init__(master, highlightthickness=0, bg="white")
        self._img_bgr: Optional[np.ndarray] = None
        self._img_tk: Optional[tk.PhotoImage] = None
        self._img_tk_scale: Optional[float] = None
        self._ov_tk: List[tk.PhotoImage] = []
        self._img_w = 0
        self._img_h = 0
        self._scale = 1.0
        self._off = (0.0, 0.0)
        self._overlays: List[Dict[str, Any]] = []
        self._anchors: List[
            Tuple[int, str, int, int]
        ] = []  # (row, 'L'|'R', x, y) in image coords
        self._drag: Optional[Tuple[int, str]] = None
        self._on_anchor_drag = on_anchor_drag
        self._on_prev_click = on_prev_click
        self._on_next_click = on_next_click
        self._nav_prev_enabled = False
        self._nav_next_enabled = False
        self._nav_prev_box: Optional[Tuple[float, float, float, float]] = None
        self._nav_next_box: Optional[Tuple[float, float, float, float]] = None
        self._redraw_pending = False
        self._redraw_after_id: Optional[str] = None
        self._drag_last_emit_t = 0.0
        self._drag_last_emit_xy: Optional[Tuple[int, int]] = None
        self._drag_min_interval_s = 1.0 / 60.0

        self.bind("<Configure>", self._on_configure)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.bind("<ButtonPress-1>", self._on_down)
        self.bind("<B1-Motion>", self._on_move)
        self.bind("<ButtonRelease-1>", self._on_up)

    def _on_configure(self, _e) -> None:
        self._request_redraw()

    def _on_destroy(self, _e) -> None:
        if self._redraw_after_id is None:
            return
        try:
            self.after_cancel(self._redraw_after_id)
        except tk.TclError:
            pass
        self._redraw_after_id = None
        self._redraw_pending = False

    def _request_redraw(self) -> None:
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self._redraw_after_id = self.after_idle(self._flush_redraw)

    def _flush_redraw(self) -> None:
        self._redraw_pending = False
        self._redraw_after_id = None
        self.redraw()

    def _photo_from_bgr(self, bgr: np.ndarray) -> tk.PhotoImage:
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise ValueError("Failed to encode BGR image as PNG.")
        b64 = base64.b64encode(buf).decode("ascii")
        return tk.PhotoImage(data=b64)

    def _photo_from_rgba(self, rgba: np.ndarray) -> tk.PhotoImage:
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise ValueError("Failed to encode RGBA image as PNG.")
        b64 = base64.b64encode(buf).decode("ascii")
        return tk.PhotoImage(data=b64)

    def set_image(self, img_bgr: Optional[np.ndarray]) -> None:
        self._img_bgr = img_bgr
        if img_bgr is not None:
            self._img_h, self._img_w = img_bgr.shape[:2]
        else:
            self._img_w = self._img_h = 0
        self._img_tk = None
        self._img_tk_scale = None
        self._request_redraw()

    def set_overlays(self, overlays: List[Dict[str, Any]]) -> None:
        self._overlays = overlays or []
        self._request_redraw()

    def set_anchors(self, anchors: List[Tuple[int, str, int, int]]) -> None:
        self._anchors = anchors
        self._request_redraw()

    def set_nav_enabled(self, has_prev: bool, has_next: bool) -> None:
        self._nav_prev_enabled = bool(has_prev)
        self._nav_next_enabled = bool(has_next)
        self._request_redraw()

    @staticmethod
    def _in_box(
        x: float, y: float, box: Optional[Tuple[float, float, float, float]]
    ) -> bool:
        if box is None:
            return False
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _draw_nav_button(
        self,
        box: Tuple[float, float, float, float],
        text: str,
        enabled: bool,
    ) -> None:
        x1, y1, x2, y2 = box
        text_color = "#ffffff" if enabled else "#000000"
        self.create_text(
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            text=text,
            fill=text_color,
            font=("TkDefaultFont", 18, "bold"),
        )

    def _fit(self) -> None:
        vw = max(1, int(self.winfo_width()))
        vh = max(1, int(self.winfo_height()))
        if self._img_bgr is None or self._img_w <= 0 or self._img_h <= 0:
            self._scale, self._off = 1.0, (0.0, 0.0)
            return
        s = min(vw / self._img_w, vh / self._img_h)
        disp_w = self._img_w * s
        disp_h = self._img_h * s
        off_x = (vw - disp_w) / 2.0
        off_y = (vh - disp_h) / 2.0
        self._scale, self._off = float(s), (float(off_x), float(off_y))

    def _view_to_img(self, mx: float, my: float) -> Optional[Tuple[float, float]]:
        if self._img_bgr is None:
            return None
        s = self._scale
        off_x, off_y = self._off
        ix = (mx - off_x) / s
        iy = (my - off_y) / s
        if ix < 0 or iy < 0 or ix > self._img_w or iy > self._img_h:
            return None
        return ix, iy

    def _img_to_view(self, ix: float, iy: float) -> Tuple[float, float]:
        s = self._scale
        off_x, off_y = self._off
        return off_x + ix * s, off_y + iy * s

    @staticmethod
    def _color_to_hex(color: Tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % color

    def _ensure_scaled_base_image(self) -> None:
        if self._img_bgr is None:
            self._img_tk = None
            self._img_tk_scale = None
            return
        s = self._scale
        if self._img_tk is not None and self._img_tk_scale == s:
            return
        w = max(1, int(round(self._img_w * s)))
        h = max(1, int(round(self._img_h * s)))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
        im = cv2.resize(self._img_bgr, (w, h), interpolation=interp)
        self._img_tk = self._photo_from_bgr(im)
        self._img_tk_scale = s

    def _draw_overlays(self) -> None:
        s = self._scale
        for o in self._overlays:
            col = self._color_to_hex(tuple(o["color"]))
            width = int(o.get("width", 2))
            if o["type"] == "rect":
                x1, y1 = self._img_to_view(o["x"], o["y"])
                x2, y2 = self._img_to_view(o["x"] + o["w"], o["y"] + o["h"])
                self.create_rectangle(x1, y1, x2, y2, outline=col, width=width)
            elif o["type"] == "mask":
                mask = o["mask"]
                if mask is None:
                    continue
                x1, y1 = self._img_to_view(o["x"], o["y"])
                h_m, w_m = mask.shape[:2]
                w_d = max(1, int(round(w_m * s)))
                h_d = max(1, int(round(h_m * s)))
                rgba = np.zeros((h_m, w_m, 4), dtype=np.uint8)
                rgba[..., 0:3] = np.array(o["color"], dtype=np.uint8)
                rgba[..., 3] = (mask > 0).astype(np.uint8) * int(o.get("alpha", 80))
                im = cv2.resize(rgba, (w_d, h_d), interpolation=cv2.INTER_NEAREST)
                tk_im = self._photo_from_rgba(im)
                self._ov_tk.append(tk_im)
                self.create_image(x1, y1, anchor="nw", image=tk_im)
            elif o["type"] == "line":
                x1, y1 = self._img_to_view(o["x1"], o["y1"])
                x2, y2 = self._img_to_view(o["x2"], o["y2"])
                self.create_line(x1, y1, x2, y2, fill=col, width=width)
            elif o["type"] == "circle":
                cx, cy = self._img_to_view(o["cx"], o["cy"])
                r = float(o["r"]) * s
                if bool(o.get("fill", False)):
                    r = max(1.5, r)
                    self.create_oval(
                        cx - r, cy - r, cx + r, cy + r, fill=col, outline="", width=0
                    )
                else:
                    self.create_oval(
                        cx - r, cy - r, cx + r, cy + r, outline=col, width=width
                    )

    def _draw_anchors(self) -> None:
        for row, side, ax, ay in self._anchors:
            x, y = self._img_to_view(ax, ay)
            r = 6
            col = "#4444ff" if side == "L" else "#ff44ff"
            self.create_oval(
                x - r, y - r, x + r, y + r, fill=col, outline="black", width=1
            )
            self.create_text(x, y - 12, text=f"{row}{side}", fill="black")

    def _draw_navigation(self) -> None:
        vw = max(1, int(self.winfo_width()))
        vh = max(1, int(self.winfo_height()))
        btn_w = 36.0
        btn_h = 84.0
        pad = 2.0
        y1 = (vh - btn_h) / 2.0
        y2 = y1 + btn_h

        self._nav_prev_box = (pad, y1, pad + btn_w, y2)
        self._nav_next_box = (vw - pad - btn_w, y1, vw - pad, y2)
        self._draw_nav_button(self._nav_prev_box, "❮", self._nav_prev_enabled)
        self._draw_nav_button(self._nav_next_box, "❯", self._nav_next_enabled)

    def redraw(self) -> None:
        self.delete("all")
        self._fit()
        self._nav_prev_box = None
        self._nav_next_box = None
        if self._img_bgr is None:
            return
        self._ov_tk = []
        self._ensure_scaled_base_image()
        off_x, off_y = self._off
        self.create_image(off_x, off_y, anchor="nw", image=self._img_tk)
        self._draw_overlays()
        self._draw_anchors()
        self._draw_navigation()

    def _on_down(self, e) -> None:
        if self._img_bgr is None:
            return
        if self._in_box(e.x, e.y, self._nav_prev_box):
            if self._nav_prev_enabled and self._on_prev_click is not None:
                self._on_prev_click()
            return
        if self._in_box(e.x, e.y, self._nav_next_box):
            if self._nav_next_enabled and self._on_next_click is not None:
                self._on_next_click()
            return

        best = None
        for row, side, ax, ay in self._anchors:
            vx, vy = self._img_to_view(ax, ay)
            d2 = (vx - e.x) ** 2 + (vy - e.y) ** 2
            if d2 <= 10 * 10 and (best is None or d2 < best[0]):
                best = (d2, row, side)
        if best:
            self._drag = (best[1], best[2])
            self._drag_last_emit_xy = None
            self._drag_last_emit_t = 0.0

    def _on_move(self, e) -> None:
        if not self._drag or self._img_bgr is None:
            return
        p = self._view_to_img(e.x, e.y)
        if not p:
            return
        ix, iy = p
        qx, qy = int(round(ix)), int(round(iy))
        if self._drag_last_emit_xy == (qx, qy):
            return
        now = perf_counter()
        if (
            self._drag_last_emit_xy is not None
            and (now - self._drag_last_emit_t) < self._drag_min_interval_s
        ):
            return
        self._drag_last_emit_xy = (qx, qy)
        self._drag_last_emit_t = now
        self._on_anchor_drag(self._drag[0], self._drag[1], ix, iy)

    def _on_up(self, _e) -> None:
        self._drag = None
        self._drag_last_emit_xy = None
