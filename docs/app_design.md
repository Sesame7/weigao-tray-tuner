# weigao-tray-tuner App Design

This document describes the design of the weigao-tray-tuner app itself.
Detection logic details are intentionally not duplicated here. The canonical
detection implementation lives in `VisionRuntime/detect/weigao_tray.py`.

## Scope

- Single-image tuning UI for the Weigao tray detector.
- Local interactive editing of anchors and parameters.
- Manual detect trigger (`Detect` button or `Ctrl+R`).
- Overlay-first result review (no separate result table).

## Canonical Source Policy

- Detector behavior and parameter semantics are sourced from:
  - `VisionRuntime/detect/weigao_tray.py`
  - `VisionRuntime/config/detect_weigao_tray.yaml`
- This repository (`weigao-tray-tuner`) is a tuning shell around that logic.
- If docs here conflict with the detector code copied from VisionRuntime, code wins.

## Runtime Architecture

- `main.py`
  - Thin entrypoint (`Tk` startup, theme setup, main loop).
- `app/controller.py`
  - App controller: image loading, run flow, state transitions.
- `ui/control_panel.py`
  - Right-side parameter panel (UI labels, parsing, write-back to params dict).
- `ui/image_view.py`
  - Left canvas: image rendering, overlays, anchor drag, prev/next hit areas.
- `core/grid_utils.py`
  - Grid ROI generation from `grid.*`.
- `core/params.py`
  - YAML load + defaults merge + save.
- `core/detector.py`
  - Bridge wrapper that instantiates copied detector.

## UI Composition

- Left panel (`ImageView`)
  - Fits image to canvas while preserving aspect ratio.
  - Draws overlays and anchor handles.
  - Provides large prev/next click zones.
- Right panel (`ControlPanel`)
  - Top buttons: `Open (Ctrl+O)` and `Detect (Ctrl+R)`.
  - Pinned section: `Grid`.
  - Tabbed sections: `A-C Rod Candidate`, `D Junction`, `E-G Final Decisions`.
  - Bottom section: `Debug Overlay`.
  - Guidance text at bottom.

## Main State Flow

1. App startup:
   - Load effective params from `config.yaml` (if present), fallback defaults from
     `config/detect_weigao_tray.yaml`.
2. Open image:
   - File dialog initial folder: current image folder (if active), else
     `data/images`, else `data`, else app base folder.
   - Load BGR via OpenCV and keep RGB for display.
3. Edit:
   - Any parameter commit or anchor drag marks params dirty.
   - Source RGB image is shown while editing.
4. Detect:
   - Call `detector.inspect_image(img_bgr, params)`.
   - Display returned overlay image.
   - Save params to `config.yaml` only if params were changed.

## Coordinates and Clamping

- All ROI and overlay coordinates are full-image pixel coordinates.
- Anchor and ROI values are clamped to image bounds.
- Row anchors enforce `x_left <= x_right`.
- Both anchors in the same row share one `y` value.

## Error Handling

- Exceptions are printed to terminal (`traceback.print_exc()`).
- UI remains usable; no modal error popup layer.
- On detect exception, canvas redraw is forced to keep view consistent.

## Non-goals

- Batch tuning workflow.
- Auto-run on every edit.
- In-app sync tooling against VisionRuntime.
- Full detector algorithm documentation copy in this repo.
