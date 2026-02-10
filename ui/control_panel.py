from __future__ import annotations

from typing import Any, Callable, Dict

import tkinter as tk
from tkinter import ttk


FieldSpec = tuple[str, ...]
SectionSpec = tuple[str, list[FieldSpec]]
LABEL_COL_MIN = 240
ENTRY_COL_MIN = 92
PANEL_WIDTH = 420
SECTION_LEFT_PAD = 6
ROW_GAP_X = 4
ENTRY_BORDER_OK = "#b8b8b8"
ENTRY_BORDER_ERR = "#d32f2f"
DEBUG_SECTION_TITLE = "Debug Overlay"

SECTION_SPECS: list[SectionSpec] = [
    (
        "Grid",
        [
            ("csv2", "Slot ROI size (W,H px)", "grid.roi.w", "grid.roi.h"),
            ("csv3", "Slots per row (r0,r1,r2)", "grid.cols_per_row"),
        ],
    ),
    (
        "A-C Rod Candidate",
        [
            ("csv3", "Yellow HSV lower (H, S, V)", "yellow_hsv.lower"),
            ("csv3", "Yellow HSV upper (H, S, V)", "yellow_hsv.upper"),
            ("int", "Mask open kernel size (px)", "morph.open_k"),
            ("int", "Mask close kernel size (px)", "morph.close_k"),
            ("int", "Mask erode kernel size (px)", "morph.erode_k"),
            ("int", "Erode iterations", "morph.erode_iter"),
            ("float", "Max tilt from vertical (deg)", "rod.vertical_tol_deg"),
            ("int", "Rod min bbox area (px^2)", "rod.min_bbox_area"),
        ],
    ),
    (
        "D Junction",
        [
            (
                "float",
                "Rod-bottom percentile of yellow pixels (%)",
                "junction.rod_bottom_percentile",
            ),
            ("int", "Junction window up from rod-bottom (px)", "junction.win_up"),
            ("int", "Junction window down from rod-bottom (px)", "junction.win_down"),
            ("int", "Strip X offset from rod center (px)", "junction.strip_offset_x"),
            ("int", "Junction strip half width (px)", "junction.strip_half_width"),
            ("int", "White mask S upper (0-255)", "white.S_max"),
            ("int", "White mask V lower (0-255)", "white.V_min"),
            ("int", "Split profile smooth half window (rows)", "split.smooth_half_win"),
            ("int", "Min rows above split", "split.min_rows_above"),
            ("int", "Min rows below split", "split.min_rows_below"),
            ("float", "Min white contrast (below-above)", "split.min_delta_white"),
        ],
    ),
    (
        "E-G Final Decisions",
        [
            ("int", "Max upward deviation from row baseline (px)", "height.delta_up"),
            ("int", "Band Y offset start from junction (px)", "band.y0"),
            ("int", "Band Y offset end from junction (px)", "band.y1"),
            ("int", "Band search half width from rod center (px)", "band.half_width"),
            ("int", "Band hue min", "band.H_low"),
            ("int", "Band hue max", "band.H_high"),
            ("int", "Band profile smooth half window (cols)", "band.smooth_half_win"),
            ("int", "Band S-profile threshold", "band.S_thr"),
            ("int", "Detected band width min (px)", "band.min_width"),
            ("int", "Detected band width max (px)", "band.max_width"),
            ("float", "Column-based reference shift (+/-px)", "offset.offset_base"),
            ("float", "Allowed band-to-reference offset (px)", "offset.offset_tol"),
        ],
    ),
    (
        DEBUG_SECTION_TITLE,
        [
            ("bool", "Show slot ROI boxes", "dbg.show_roi"),
            ("bool", "Show cleaned yellow mask", "dbg.show_mask"),
            ("bool", "Show junction left/right strips", "dbg.show_strips"),
            ("bool", "Show band search ROI", "dbg.show_band"),
        ],
    ),
]

PINNED_SECTIONS = ("Grid",)


def _getp(d: Any, path: str) -> Any:
    cur = d
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _setp(d: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


class ControlPanel:
    def __init__(
        self,
        parent,
        params: Dict[str, Any],
        on_params_change: Callable[[], None],
        on_open: Callable[[], None],
        on_run: Callable[[], None],
    ):
        self.params = params
        self._on_params_change = on_params_change
        self._tk_vars: list[tk.Variable] = []

        self.frame = ttk.Frame(parent, width=PANEL_WIDTH)
        self.frame.grid_propagate(False)

        self.guidance = tk.StringVar(value="Open an image to start (Ctrl+O).")
        self.btn_detect: ttk.Button | None = None

        self._build_panel(on_open, on_run)

    def set_guidance(self, text: str) -> None:
        self.guidance.set(text)

    def set_detect_enabled(self, enabled: bool) -> None:
        if self.btn_detect is not None:
            self.btn_detect.configure(state=("normal" if enabled else "disabled"))

    def _build_panel(
        self,
        on_open: Callable[[], None],
        on_run: Callable[[], None],
    ) -> None:
        r = 0
        btns = ttk.Frame(self.frame)
        btns.grid(sticky="ew", row=r, column=0, columnspan=2)
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        ttk.Button(btns, text="Open (Ctrl+O)", command=on_open).grid(
            sticky="ew", row=0, column=0, padx=(0, 2)
        )
        self.btn_detect = ttk.Button(
            btns, text="Detect (Ctrl+R)", command=on_run, state="disabled"
        )
        self.btn_detect.grid(sticky="ew", row=0, column=1, padx=(2, 0))
        r += 1

        rows_by_parent: Dict[Any, int] = {}

        def next_row(parent: Any) -> int:
            row = rows_by_parent.get(parent, 0)
            rows_by_parent[parent] = row + 1
            return row

        def add_section(title: str) -> ttk.Frame:
            nonlocal r
            section = ttk.Frame(self.frame)
            section.grid(sticky="ew", row=r, column=0, columnspan=2, pady=(4, 0))
            section.columnconfigure(0, weight=1)

            ttk.Label(section, text=title, font=("TkDefaultFont", 10, "bold")).grid(
                sticky="w", row=0, column=0
            )
            body = ttk.Frame(section)
            body.grid(
                sticky="ew", row=1, column=0, padx=(SECTION_LEFT_PAD, 0), pady=(2, 0)
            )
            body.columnconfigure(0, weight=0, minsize=LABEL_COL_MIN)
            body.columnconfigure(1, weight=1, minsize=ENTRY_COL_MIN)

            r += 1
            return body

        def add_tab(notebook: ttk.Notebook, title: str) -> ttk.Frame:
            tab = ttk.Frame(notebook, padding=(6, 4))
            tab.columnconfigure(0, weight=0, minsize=LABEL_COL_MIN)
            tab.columnconfigure(1, weight=1, minsize=ENTRY_COL_MIN)
            notebook.add(tab, text=title)
            return tab

        def add_labeled_entry(
            parent: Any,
            label: str,
            value: str,
            on_change: Callable[[str], None],
            get_display: Callable[[], str] | None = None,
        ) -> None:
            row = next_row(parent)
            var = tk.StringVar(value=value)
            last_value: Dict[str, str] = {"text": value}
            ttk.Label(parent, text=label, justify="left").grid(
                sticky="w", row=row, column=0, padx=(0, ROW_GAP_X)
            )
            entry = tk.Entry(
                parent,
                textvariable=var,
                width=13,
                justify="right",
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground=ENTRY_BORDER_OK,
                highlightcolor=ENTRY_BORDER_OK,
            )
            entry.grid(sticky="e", row=row, column=1)

            def set_entry_valid(valid: bool) -> None:
                border = ENTRY_BORDER_OK if valid else ENTRY_BORDER_ERR
                fg = "#000000" if valid else "#b71c1c"
                entry.configure(
                    highlightbackground=border, highlightcolor=border, fg=fg
                )

            def commit(*_):
                text = var.get().strip()
                if text == last_value["text"]:
                    set_entry_valid(True)
                    return
                try:
                    on_change(text)
                    new_text = get_display() if get_display is not None else text
                    last_value["text"] = new_text
                    var.set(new_text)
                    set_entry_valid(True)
                    self._on_params_change()
                except Exception:
                    set_entry_valid(False)

            entry.bind("<Return>", commit)
            entry.bind("<KP_Enter>", commit)
            entry.bind("<FocusOut>", commit)

        def create_bool_checkbutton(
            parent: Any, label: str, path: str
        ) -> ttk.Checkbutton:
            var = tk.BooleanVar(value=bool(_getp(self.params, path)))
            self._tk_vars.append(var)
            cb = ttk.Checkbutton(
                parent,
                text=label,
                variable=var,
                command=lambda v=var, p=path: (
                    _setp(self.params, p, bool(v.get())),
                    self._on_params_change(),
                ),
            )
            return cb

        def add_bool(parent: Any, label: str, path: str) -> None:
            row = next_row(parent)
            create_bool_checkbutton(parent, label, path).grid(
                sticky="w", row=row, column=0, columnspan=2
            )

        def add_field(parent: Any, spec: FieldSpec) -> None:
            kind, label, *paths = spec

            if kind == "bool":
                add_bool(parent, label, paths[0])
                return

            if kind == "int":
                path = paths[0]
                add_labeled_entry(
                    parent,
                    label,
                    str(_getp(self.params, path)),
                    lambda s: _setp(self.params, path, int(float(s))),
                    lambda: str(int(_getp(self.params, path))),
                )
                return

            if kind == "float":
                path = paths[0]
                add_labeled_entry(
                    parent,
                    label,
                    str(_getp(self.params, path)),
                    lambda s: _setp(self.params, path, float(s)),
                    lambda: str(float(_getp(self.params, path))),
                )
                return

            if kind == "csv2":
                path_a, path_b = paths

                def apply_csv2(s: str) -> None:
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) != 2:
                        raise ValueError("csv2 expects 2 values")
                    a, b = (int(float(parts[0])), int(float(parts[1])))
                    _setp(self.params, path_a, a)
                    _setp(self.params, path_b, b)

                add_labeled_entry(
                    parent,
                    label,
                    f"{int(_getp(self.params, path_a))},{int(_getp(self.params, path_b))}",
                    apply_csv2,
                    lambda: (
                        f"{int(_getp(self.params, path_a))},{int(_getp(self.params, path_b))}"
                    ),
                )
                return

            if kind == "csv3":
                path = paths[0]

                def apply_csv3(s: str) -> None:
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) != 3:
                        raise ValueError("csv3 expects 3 values")
                    vals = [int(float(p)) for p in parts]
                    _setp(self.params, path, vals)

                add_labeled_entry(
                    parent,
                    label,
                    ",".join(str(int(x)) for x in _getp(self.params, path)),
                    apply_csv3,
                    lambda: ",".join(str(int(x)) for x in _getp(self.params, path)),
                )
                return

            raise ValueError(f"Unsupported field kind: {kind}")

        def render_fields(
            parent: Any, fields: list[FieldSpec], bool_columns: int | None = None
        ) -> None:
            if bool_columns is not None and all(spec[0] == "bool" for spec in fields):
                for i, spec in enumerate(fields):
                    _kind, label, path = spec
                    row_i = i // bool_columns
                    col_i = i % bool_columns
                    create_bool_checkbutton(parent, label, path).grid(
                        sticky="w", row=row_i, column=col_i, padx=(0, 8), pady=(0, 2)
                    )
                return
            for field in fields:
                add_field(parent, field)

        specs_by_title: Dict[str, list[FieldSpec]] = {
            title: fields for title, fields in SECTION_SPECS
        }

        for pinned in PINNED_SECTIONS:
            if pinned in specs_by_title:
                fields = specs_by_title[pinned]
                if pinned == "Grid":
                    sec = ttk.Frame(self.frame)
                    sec.grid(
                        sticky="ew",
                        row=r,
                        column=0,
                        columnspan=2,
                        padx=(SECTION_LEFT_PAD, 0),
                        pady=(4, 0),
                    )
                    sec.columnconfigure(0, weight=0, minsize=LABEL_COL_MIN)
                    sec.columnconfigure(1, weight=1, minsize=ENTRY_COL_MIN)
                    r += 1
                else:
                    sec = add_section(pinned)
                render_fields(sec, fields)

        stage_specs = [
            (title, fields)
            for title, fields in SECTION_SPECS
            if title not in PINNED_SECTIONS and title != DEBUG_SECTION_TITLE
        ]
        if stage_specs:
            tabs = ttk.Notebook(self.frame)
            tabs.grid(sticky="nsew", row=r, column=0, columnspan=2, pady=(6, 2))
            tabs_row = r
            r += 1

            for title, fields in stage_specs:
                tab = add_tab(tabs, title)
                render_fields(tab, fields)

            self.frame.rowconfigure(tabs_row, weight=1)

        if DEBUG_SECTION_TITLE in specs_by_title:
            fields = specs_by_title[DEBUG_SECTION_TITLE]
            sec = ttk.Frame(self.frame)
            sec.grid(
                sticky="ew",
                row=r,
                column=0,
                columnspan=2,
                padx=(SECTION_LEFT_PAD, 2),
                pady=(4, 0),
            )
            for i in range(2):
                sec.columnconfigure(i, weight=1)
            r += 1
            render_fields(sec, fields, bool_columns=2)

        ttk.Separator(self.frame, orient="horizontal").grid(
            sticky="ew", row=r, column=0, columnspan=2, pady=(8, 6)
        )
        r += 1

        guidance_label = ttk.Label(
            self.frame, textvariable=self.guidance, justify="left"
        )
        guidance_label.grid(sticky="ew", row=r, column=0, columnspan=2, pady=(6, 0))

        def _update_guidance_wraplength(_event) -> None:
            width = int(self.frame.winfo_width())
            guidance_label.configure(wraplength=max(200, width - 8))

        self.frame.bind("<Configure>", _update_guidance_wraplength, add="+")
        self.frame.columnconfigure(0, weight=0)
        self.frame.columnconfigure(1, weight=1)
