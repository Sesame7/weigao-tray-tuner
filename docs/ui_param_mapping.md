# UI Parameter Mapping

This file maps weigao-tray-tuner UI labels to the current upstream detector keys.

Reference implementation:

- `ui/control_panel.py` (UI labels and binding)
- `detect/weigao_tray.py` (actual parameter consumption)
- `config/detect_weigao_tray.yaml` (default values and structure)

## Slot Layout

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Slot ROI size (W,H px) | `slot_layout.slot_roi.width_px`, `slot_layout.slot_roi.height_px` | csv2 int | ROI generation for all stages |
| Slots per row (comma-separated) | `slot_layout.slots_per_row` | csvN int | ROI generation and color-band alignment normalization |

Anchors are edited by dragging in the image view:

- `slot_layout.left_anchors[*].x/y`
- `slot_layout.right_anchors[*].x/y`

## Stem

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Stem HSV lower (H, S, V) | `stem.hsv.lower` | csv3 int | Stem mask threshold |
| Stem HSV upper (H, S, V) | `stem.hsv.upper` | csv3 int | Stem mask threshold |
| Stem min area (px) | `stem.min_area_px` | int | Stem presence gate |
| Stem bottom cutoff percentile (%) | `stem.bottom_exclude_percentile` | float | Stem anchor computation |

## Junction Line

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Junction search window from stem anchor (up/down px) | `junction_line.search_band.y_from_anchor_px.up`, `junction_line.search_band.y_from_anchor_px.down` | csv2 int | Junction search band |
| Junction strip radius (outer / inner-exclude px) | `junction_line.search_band.x_radius_from_stem_center_px.outer`, `junction_line.search_band.x_radius_from_stem_center_px.inner_exclude` | csv2 int | Left/right strip spans |
| Row<i> white threshold (Smax,Vmin) | `junction_line.white_by_row.<i>.s_max`, `junction_line.white_by_row.<i>.v_min` | csv2 int | Row i white mask (dynamic by row count) |
| Split smooth window (rows, odd) | `junction_line.split.smooth_window_rows` | int | White profile smoothing |
| Min split rows (above,below) | `junction_line.split.min_split_rows.above`, `junction_line.split.min_split_rows.below` | csv2 int | Split validity |
| Height max deviation from row median (px) | `junction_line.height_check.max_deviation_px` | int | Junction-line height gate |

## Color Band

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Color-band Y range below junction (from/to px) | `color_band.search_window.y_below_junction_line_px.from`, `color_band.search_window.y_below_junction_line_px.to` | csv2 int | Color-band search window |
| Band X radius from stem center (px) | `color_band.search_window.x_radius_from_stem_center_px` | int | Color-band search window |
| Band hue range (low,high) | `color_band.color_response.hue_range.low`, `color_band.color_response.hue_range.high` | csv2 int | Hue gate |
| Band saturation threshold | `color_band.color_response.saturation.threshold` | int | Band response threshold |
| Band saturation smooth window (cols, odd) | `color_band.color_response.saturation.smooth_window_cols` | int | Band response smoothing |
| Band width check (min,max px) | `color_band.width_check_px.min`, `color_band.width_check_px.max` | csv2 int | Span validity |
| Alignment reference X shift (px) | `color_band.alignment_check.reference_x_shift_px` | float | Reference center shift |
| Alignment tolerance (px) | `color_band.alignment_check.tolerance_px` | float | Alignment gate |

## Debug Overlay

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Show slot ROI boxes | `debug_overlay.slot_layout` | bool | App-side ROI boxes + detector guide convention |
| Show stem mask | `debug_overlay.stem_mask` | bool | Stem mask overlay |
| Show junction strips | `debug_overlay.junction_line` | bool | Junction search strip guides |
| Show color band ROI | `debug_overlay.color_band` | bool | Color-band search guides |

## Hidden Params (Not Exposed in UI)

- `result_overlay.ok_bgr`
- `result_overlay.ng_bgr`
- `result_overlay.stroke_width_px`
