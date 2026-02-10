# Detection Reference and Compatibility

This document defines how weigao-tray-tuner references detector behavior without
becoming a second source of truth.

## Canonical Detector Source

The detector behavior used by this app is copied from the main project:

- Main project path: `<VISION_RUNTIME_ROOT>`
- Canonical files:
  - `VisionRuntime/detect/weigao_tray.py`
  - `VisionRuntime/config/detect_weigao_tray.yaml`

In this repo, the runtime copies are:

- `detect/weigao_tray.py`
- `config/detect_weigao_tray.yaml`

If this document differs from copied code, copied code is authoritative.

## weigao-tray-tuner Detection Adapter

weigao-tray-tuner does not implement detector stages directly. The call chain is:

1. `app/controller.py` calls `detector.inspect_image(img_bgr, params)`.
2. `core/detector.py` instantiates `WeigaoTrayDetector`.
3. `detect/weigao_tray.py` runs the actual A-G stage logic and returns overlay.

## Stage Contract (A-G Summary)

This is a compact contract for UI and parameter mapping only.

- Stage A: yellow mask clean, `ROD_NOT_FOUND`
- Stage B: center + tilt from vertical, `ROD_ANGLE_NG`
- Stage C: bbox area gate, `ROD_BBOX_AREA_NG`
- Stage D: junction split/contrast, `JUNCTION_NOT_FOUND`
- Stage E: row baseline height decision, `HEIGHT_NG`
- Stage F: band search in ROI under junction, `BAND_NOT_FOUND`
- Stage G: band offset against column-dependent reference, `OFFSET_NG`

First failing slot/code is returned as image-level NG result code.

## Overlay Contract Used by UI

- Overlay image is returned by detector and shown directly in `ImageView`.
- Grid overlay (`dbg.show_roi`) is additionally rendered by app-side helpers in
  editor mode.
- Debug toggles currently exposed in UI:
  - `dbg.show_roi`
  - `dbg.show_mask`
  - `dbg.show_strips`
  - `dbg.show_band`

## What Must Stay Aligned

After sync from VisionRuntime, verify these before release:

1. NG code names expected by UI/docs still exist.
2. Parameter keys used by `ui/control_panel.py` still exist and keep compatible types.
3. Debug toggle semantics did not change.
4. Any newly added detector parameter is either exposed in UI or intentionally
   documented as hidden.

## Deliberate Non-goal

This file is not a full algorithm specification. Full algorithm detail should be
read from `detect/weigao_tray.py` in the synced detector copy.

