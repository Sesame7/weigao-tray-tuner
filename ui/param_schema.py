from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


FieldKind = Literal["bool", "int", "float", "csv2", "csv2f", "csv3", "csvN"]
DEBUG_SECTION_TITLE = "Debug Overlay"
PINNED_SECTIONS = ("Slot Layout",)


@dataclass(frozen=True)
class FieldSpec:
    kind: FieldKind
    label: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class SectionSpec:
    title: str
    fields: tuple[FieldSpec, ...]


def row_count_from_params(params: Mapping[str, Any]) -> int:
    slot_layout = params.get("slot_layout")
    if not isinstance(slot_layout, Mapping):
        return 3
    rows = slot_layout.get("slots_per_row")
    if not isinstance(rows, list) or len(rows) <= 0:
        return 3
    return len(rows)


def build_section_specs(params: Mapping[str, Any]) -> list[SectionSpec]:
    rows = row_count_from_params(params)
    junction_white_fields = tuple(
        FieldSpec(
            kind="csv2",
            label=f"Row{i} white threshold (Smax,Vmin)",
            paths=(
                f"junction_line.white_by_row.{i}.s_max",
                f"junction_line.white_by_row.{i}.v_min",
            ),
        )
        for i in range(rows)
    )

    return [
        SectionSpec(
            title="Slot Layout",
            fields=(
                FieldSpec(
                    kind="csv2",
                    label="Slot ROI size (W,H px)",
                    paths=(
                        "slot_layout.slot_roi.width_px",
                        "slot_layout.slot_roi.height_px",
                    ),
                ),
                FieldSpec(
                    kind="csvN",
                    label="Slots per row (comma-separated)",
                    paths=("slot_layout.slots_per_row",),
                ),
            ),
        ),
        SectionSpec(
            title="Stem",
            fields=(
                FieldSpec("csv3", "Stem HSV lower (H, S, V)", ("stem.hsv.lower",)),
                FieldSpec("csv3", "Stem HSV upper (H, S, V)", ("stem.hsv.upper",)),
                FieldSpec("int", "Stem min area (px)", ("stem.min_area_px",)),
                FieldSpec(
                    "float",
                    "Stem bottom cutoff percentile (%)",
                    ("stem.bottom_exclude_percentile",),
                ),
            ),
        ),
        SectionSpec(
            title="Junction Line",
            fields=(
                FieldSpec(
                    "csv2",
                    "Junction Y window (up/down px)",
                    (
                        "junction_line.search_band.y_from_anchor_px.up",
                        "junction_line.search_band.y_from_anchor_px.down",
                    ),
                ),
                FieldSpec(
                    "csv2",
                    "Junction strip radius (outer / inner-exclude px)",
                    (
                        "junction_line.search_band.x_radius_from_stem_center_px.outer",
                        "junction_line.search_band.x_radius_from_stem_center_px.inner_exclude",
                    ),
                ),
                *junction_white_fields,
                FieldSpec(
                    "csv2",
                    "Min split rows (above,below)",
                    (
                        "junction_line.split.min_split_rows.above",
                        "junction_line.split.min_split_rows.below",
                    ),
                ),
                FieldSpec(
                    "int",
                    "Height max deviation from row median (px)",
                    ("junction_line.height_check.max_deviation_px",),
                ),
            ),
        ),
        SectionSpec(
            title="Color Band",
            fields=(
                FieldSpec(
                    "csv2",
                    "Band Y below junction (from/to px)",
                    (
                        "color_band.search_window.y_below_junction_line_px.from",
                        "color_band.search_window.y_below_junction_line_px.to",
                    ),
                ),
                FieldSpec(
                    "int",
                    "Band X radius from stem center (px)",
                    ("color_band.search_window.x_radius_from_stem_center_px",),
                ),
                FieldSpec(
                    "csv2",
                    "Band hue range (low,high)",
                    (
                        "color_band.color_response.hue_range.low",
                        "color_band.color_response.hue_range.high",
                    ),
                ),
                FieldSpec(
                    "int",
                    "Band saturation threshold",
                    ("color_band.color_response.saturation.threshold",),
                ),
                FieldSpec(
                    "csv2",
                    "Band width check (min,max px)",
                    ("color_band.width_check_px.min", "color_band.width_check_px.max"),
                ),
                FieldSpec(
                    "csv2f",
                    "Alignment check (reference shift,tolerance px)",
                    (
                        "color_band.alignment_check.reference_x_shift_px",
                        "color_band.alignment_check.tolerance_px",
                    ),
                ),
            ),
        ),
        SectionSpec(
            title=DEBUG_SECTION_TITLE,
            fields=(
                FieldSpec(
                    "bool", "Show slot ROI boxes", ("debug_overlay.slot_layout",)
                ),
                FieldSpec("bool", "Show stem mask", ("debug_overlay.stem_mask",)),
                FieldSpec(
                    "bool",
                    "Show junction strips",
                    ("debug_overlay.junction_line",),
                ),
                FieldSpec("bool", "Show color band ROI", ("debug_overlay.color_band",)),
            ),
        ),
    ]
