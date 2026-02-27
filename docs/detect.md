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
3. `detect/weigao_tray.py` runs the actual stage logic and returns overlay.

## Stage Contract (Current)

This is a compact contract for UI and parameter mapping only.

- Stem stage: locate stem mask/center/anchor, `STEM_NOT_FOUND_NG`
- Junction-line locate stage: find split line, `JUNCTION_LINE_NOT_FOUND_NG`
- Junction-line height stage: row-median consistency check, `JUNCTION_LINE_HEIGHT_NG`
- Color-band locate stage: band span in search window, `COLOR_BAND_NOT_FOUND_NG`
- Color-band alignment stage: compare to column-based reference, `COLOR_BAND_ALIGNMENT_NG`

First failing slot/code is returned as image-level NG result code.

## Overlay Contract Used by UI

- Overlay image is returned by detector and shown directly in `ImageView`.
- Grid overlay (`debug_overlay.slot_layout`) is additionally rendered by app-side
  helpers in
  editor mode.
- Debug toggles currently exposed in UI:
  - `debug_overlay.slot_layout`
  - `debug_overlay.stem_mask`
  - `debug_overlay.junction_line`
  - `debug_overlay.color_band`

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

