from __future__ import annotations

from typing import Any, Callable, Dict

import tkinter as tk
from tkinter import ttk

from ui.param_schema import (
    DEBUG_SECTION_TITLE,
    FieldSpec,
    SectionSpec,
    build_section_specs,
)
from ui.param_state import ParamState

LABEL_COL_MIN = 240
ENTRY_COL_MIN = 92
PANEL_WIDTH = 360
SECTION_LEFT_PAD = 6
ROW_GAP_X = 4
ENTRY_BORDER_OK = "#b8b8b8"
ENTRY_BORDER_ERR = "#d32f2f"
NON_COLLAPSIBLE_SECTIONS = {"Slot Layout"}


class ControlPanel:
    def __init__(
        self,
        parent,
        params: Dict[str, Any],
        on_params_change: Callable[[], None],
        on_open: Callable[[], None],
        on_run: Callable[[], None],
    ):
        self.state = ParamState(params)
        self._on_params_change = on_params_change
        self._on_open = on_open
        self._on_run = on_run
        self._detect_enabled = False
        self._tk_vars: list[tk.Variable] = []
        self._section_specs: list[SectionSpec] = []
        self._section_open: Dict[str, bool] = {}

        self.frame = ttk.Frame(parent, width=PANEL_WIDTH)
        self.frame.grid_propagate(False)

        self.guidance = tk.StringVar(value="Open an image to start (Ctrl+O).")
        self.btn_detect: ttk.Button | None = None

        self._rebuild_panel()

    @property
    def params(self) -> Dict[str, Any]:
        return self.state.params

    def set_guidance(self, text: str) -> None:
        self.guidance.set(text)

    def set_detect_enabled(self, enabled: bool) -> None:
        self._detect_enabled = bool(enabled)
        if self.btn_detect is not None:
            self.btn_detect.configure(state=("normal" if enabled else "disabled"))

    def _rebuild_panel(self) -> None:
        self._section_specs = build_section_specs(self.state.params)
        self._tk_vars.clear()
        for child in self.frame.winfo_children():
            child.destroy()
        self._build_panel(self._on_open, self._on_run)

    def _build_panel(
        self,
        on_open: Callable[[], None],
        on_run: Callable[[], None],
    ) -> None:
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        r = 0
        btns = ttk.Frame(self.frame)
        btns.grid(sticky="ew", row=r, column=0)
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        ttk.Button(btns, text="Open (Ctrl+O)", command=on_open).grid(
            sticky="ew", row=0, column=0, padx=(0, 2)
        )
        self.btn_detect = ttk.Button(
            btns,
            text="Detect (Ctrl+R)",
            command=on_run,
            state=("normal" if self._detect_enabled else "disabled"),
        )
        self.btn_detect.grid(sticky="ew", row=0, column=1, padx=(2, 0))
        r += 1

        content = ttk.Frame(self.frame)
        content.grid(sticky="nsew", row=r, column=0, pady=(4, 0))
        content.columnconfigure(0, weight=1)
        r += 1

        rows_by_parent: Dict[Any, int] = {}
        content_row = 0

        def next_row(parent: Any) -> int:
            row = rows_by_parent.get(parent, 0)
            rows_by_parent[parent] = row + 1
            return row

        def add_labeled_entry(
            parent: Any,
            label: str,
            value: str,
            on_change: Callable[[str], bool],
            get_display: Callable[[], str],
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
                    highlightbackground=border,
                    highlightcolor=border,
                    fg=fg,
                )

            def commit(*_) -> None:
                text = var.get().strip()
                if text == last_value["text"]:
                    set_entry_valid(True)
                    return
                try:
                    should_rebuild = on_change(text)
                    new_text = get_display()
                    last_value["text"] = new_text
                    var.set(new_text)
                    set_entry_valid(True)
                    self._on_params_change()
                    if should_rebuild:
                        self._rebuild_panel()
                except Exception:
                    set_entry_valid(False)

            entry.bind("<Return>", commit)
            entry.bind("<KP_Enter>", commit)
            entry.bind("<FocusOut>", commit)

        def create_bool_checkbutton(parent: Any, field: FieldSpec) -> ttk.Checkbutton:
            path = field.paths[0]
            var = tk.BooleanVar(value=self.state.get_bool(path))
            self._tk_vars.append(var)
            cb = ttk.Checkbutton(
                parent,
                text=field.label,
                variable=var,
                command=lambda v=var, p=path: (
                    self.state.set_bool(p, bool(v.get())),
                    self._on_params_change(),
                ),
            )
            return cb

        def add_field(parent: Any, field: FieldSpec) -> None:
            if field.kind == "bool":
                row = next_row(parent)
                create_bool_checkbutton(parent, field).grid(
                    sticky="w", row=row, column=0, columnspan=2
                )
                return
            add_labeled_entry(
                parent,
                field.label,
                self.state.display_text(field),
                lambda s, f=field: self.state.update_from_text(f, s),
                lambda f=field: self.state.display_text(f),
            )

        def render_fields(
            parent: Any,
            fields: tuple[FieldSpec, ...],
            bool_columns: int | None = None,
        ) -> None:
            if bool_columns is not None and all(
                field.kind == "bool" for field in fields
            ):
                for i, field in enumerate(fields):
                    row_i = i // bool_columns
                    col_i = i % bool_columns
                    create_bool_checkbutton(parent, field).grid(
                        sticky="w", row=row_i, column=col_i, padx=(0, 8), pady=(0, 2)
                    )
                return
            for field in fields:
                add_field(parent, field)

        def add_collapsible_section(
            title: str,
            fields: tuple[FieldSpec, ...],
            *,
            bool_columns: int | None = None,
            default_open: bool = True,
        ) -> None:
            nonlocal content_row
            section = ttk.Frame(content)
            section.grid(sticky="ew", row=content_row, column=0, pady=(2, 0))
            section.columnconfigure(0, weight=1)
            content_row += 1

            is_open = self._section_open.get(title, default_open)
            self._section_open[title] = bool(is_open)

            header_label = ttk.Label(section, cursor="hand2")
            header_label.grid(sticky="ew", row=0, column=0)

            body = ttk.Frame(section)
            body.grid(
                sticky="ew",
                row=1,
                column=0,
                padx=(SECTION_LEFT_PAD, 0),
                pady=(1, 0),
            )
            body.columnconfigure(0, weight=0, minsize=LABEL_COL_MIN)
            body.columnconfigure(1, weight=1, minsize=ENTRY_COL_MIN)
            render_fields(body, fields, bool_columns=bool_columns)

            def _apply_state() -> None:
                open_now = bool(self._section_open.get(title, True))
                header_label.configure(
                    text=(f"[-] {title}" if open_now else f"[+] {title}")
                )
                if open_now:
                    body.grid()
                else:
                    body.grid_remove()

            def _toggle() -> None:
                self._section_open[title] = not bool(
                    self._section_open.get(title, True)
                )
                _apply_state()

            header_label.bind("<Button-1>", lambda _e: _toggle(), add="+")
            _apply_state()

        def add_plain_section(
            fields: tuple[FieldSpec, ...],
            *,
            bool_columns: int | None = None,
        ) -> None:
            nonlocal content_row
            body = ttk.Frame(content)
            body.grid(sticky="ew", row=content_row, column=0, pady=(4, 0))
            body.columnconfigure(0, weight=0, minsize=LABEL_COL_MIN)
            body.columnconfigure(1, weight=1, minsize=ENTRY_COL_MIN)
            content_row += 1
            render_fields(body, fields, bool_columns=bool_columns)

        debug_fields: tuple[FieldSpec, ...] | None = None
        for section in self._section_specs:
            if section.title == DEBUG_SECTION_TITLE:
                debug_fields = section.fields
                continue
            bool_columns = 2 if section.title == DEBUG_SECTION_TITLE else None
            if section.title in NON_COLLAPSIBLE_SECTIONS:
                add_plain_section(section.fields, bool_columns=bool_columns)
                continue
            add_collapsible_section(
                section.title,
                section.fields,
                bool_columns=bool_columns,
                default_open=True,
            )

        ttk.Separator(self.frame, orient="horizontal").grid(
            sticky="ew", row=r, column=0, pady=(8, 2)
        )
        r += 1

        guidance_label = ttk.Label(
            self.frame, textvariable=self.guidance, justify="left"
        )
        guidance_label.grid(sticky="ew", row=r, column=0, pady=(0, 2))
        r += 1

        if debug_fields is not None:
            debug_frame = ttk.Frame(self.frame)
            debug_frame.grid(sticky="ew", row=r, column=0, pady=(0, 0))
            debug_frame.columnconfigure(0, weight=1)
            debug_frame.columnconfigure(1, weight=1)
            render_fields(debug_frame, debug_fields, bool_columns=2)

        def _update_guidance_wraplength(_event) -> None:
            width = int(self.frame.winfo_width())
            guidance_label.configure(wraplength=max(200, width - 8))

        self.frame.bind("<Configure>", _update_guidance_wraplength, add="+")
