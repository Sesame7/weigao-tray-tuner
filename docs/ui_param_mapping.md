# UI Parameter Mapping

This file maps weigao-tray-tuner UI labels to detector parameter keys and stage usage.

Reference implementation:

- `ui/control_panel.py` (UI labels and binding)
- `detect/weigao_tray.py` (actual parameter consumption)

## Grid

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Slot ROI size (W,H px) | `grid.roi.w`, `grid.roi.h` | csv2 int | ROI generation for all stages |
| Slots per row (r0,r1,r2) | `grid.cols_per_row` | csv3 int | ROI generation and Stage G column normalization |

## A-C Rod Candidate

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Yellow HSV lower (H, S, V) | `yellow_hsv.lower` | csv3 int | Stage A mask threshold |
| Yellow HSV upper (H, S, V) | `yellow_hsv.upper` | csv3 int | Stage A mask threshold |
| Mask open kernel size (px) | `morph.open_k` | int | Stage A morphology |
| Mask close kernel size (px) | `morph.close_k` | int | Stage A morphology |
| Mask erode kernel size (px) | `morph.erode_k` | int | Stage A morphology |
| Erode iterations | `morph.erode_iter` | int | Stage A morphology |
| Max tilt from vertical (deg) | `rod.vertical_tol_deg` | float | Stage B angle gate |
| Rod min bbox area (px^2) | `rod.min_bbox_area` | int | Stage C bbox gate |

## D Junction

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Rod-bottom percentile of yellow pixels (%) | `junction.rod_bottom_percentile` | float | Stage D rod-bottom estimate |
| Junction window up from rod-bottom (px) | `junction.win_up` | int | Stage D vertical window |
| Junction window down from rod-bottom (px) | `junction.win_down` | int | Stage D vertical window |
| Strip X offset from rod center (px) | `junction.strip_offset_x` | int | Stage D strip placement |
| Junction strip half width (px) | `junction.strip_half_width` | int | Stage D strip placement |
| White mask S upper (0-255) | `white.S_max` | int | Stage D white mask |
| White mask V lower (0-255) | `white.V_min` | int | Stage D white mask |
| Split profile smooth half window (rows) | `split.smooth_half_win` | int | Stage D profile smoothing |
| Min rows above split | `split.min_rows_above` | int | Stage D split validity |
| Min rows below split | `split.min_rows_below` | int | Stage D split validity |
| Min white contrast (below-above) | `split.min_delta_white` | float | Stage D split contrast threshold |

## E-G Final Decisions

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Max upward deviation from row baseline (px) | `height.delta_up` | int | Stage E baseline threshold |
| Band Y offset start from junction (px) | `band.y0` | int | Stage F band ROI |
| Band Y offset end from junction (px) | `band.y1` | int | Stage F band ROI |
| Band search half width from rod center (px) | `band.half_width` | int | Stage F band ROI |
| Band hue min | `band.H_low` | int | Stage F hue gate |
| Band hue max | `band.H_high` | int | Stage F hue gate |
| Band profile smooth half window (cols) | `band.smooth_half_win` | int | Stage F profile smoothing |
| Band S-profile threshold | `band.S_thr` | int | Stage F band run threshold |
| Detected band width min (px) | `band.min_width` | int | Stage F width validity |
| Detected band width max (px) | `band.max_width` | int | Stage F width validity |
| Column-based reference shift (+/-px) | `offset.offset_base` | float | Stage G reference line |
| Allowed band-to-reference offset (px) | `offset.offset_tol` | float | Stage G final gate |

## Debug Overlay

| UI label | Key(s) | Type | Stage/use |
|---|---|---|---|
| Show slot ROI boxes | `dbg.show_roi` | bool | App-side ROI box preview + detector debug conventions |
| Show cleaned yellow mask | `dbg.show_mask` | bool | Stage A debug mask overlay |
| Show junction left/right strips | `dbg.show_strips` | bool | Stage D strip debug |
| Show band search ROI | `dbg.show_band` | bool | Stage F band ROI debug |

## Hidden (Not Exposed in Current UI)

These keys exist in detector config but are not currently shown in weigao-tray-tuner UI:

- `denoise.k`, `denoise.sigma`
- `morph.keep_largest`
- `band.use_median`
- `overlay.ok_bgr`, `overlay.ng_bgr`, `overlay.line_width`
- grid anchor arrays (`grid.anchor_left`, `grid.anchor_right`) are edited via drag, not text field

If any hidden key becomes operationally important for tuning, expose it in
`ui/control_panel.py` and update this file.
