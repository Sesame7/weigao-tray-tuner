# weigao-tray-tuner

Single-image tuning UI for the Weigao tray detector.

## Purpose

weigao-tray-tuner is the tuning shell for detector logic synced from VisionRuntime.
This repo focuses on:

- visual parameter adjustment
- anchor placement
- overlay-based inspection review

Detector algorithm logic remains in the synced detector files under `detect/`.

## Sync Workflow (Manual Copy)

Manually copy from VisionRuntime:

- `VisionRuntime/config/detect_weigao_tray.yaml` -> `weigao-tray-tuner/config/detect_weigao_tray.yaml`
- `VisionRuntime/detect/*` -> `weigao-tray-tuner/detect/*`
- optional samples -> `weigao-tray-tuner/data/images/*`

## Launch

```powershell
pip install -r requirements.txt
python main.py
```

## Packaging (PyInstaller onedir)

Build a folder-based package (`onedir`):

```powershell
pip install pyinstaller
pyinstaller --noconfirm --clean --windowed --name weigao-tray-tuner main.py
```

After build, copy the detector default config next to the generated exe:

- source: `config/detect_weigao_tray.yaml`
- target: `dist/weigao-tray-tuner/detect_weigao_tray.yaml`

Run:

- `dist/weigao-tray-tuner/weigao-tray-tuner.exe`

Notes:

- Keep the whole `dist/weigao-tray-tuner/` folder together (do not move only the exe).
- The packaged app reads/writes `config.yaml` in the same folder as the exe.
- The packaged app expects `detect_weigao_tray.yaml` in the same folder as the exe.
- `onedir` output includes runtime dependencies under `_internal/` (OpenCV / NumPy / Tk); this is normal.

## UI Workflow

1. Click `Open` to load an image.
2. Drag 6 anchors (`0/1/2` x `L/R`) and adjust parameters.
3. Click `Detect` to run inspection and update overlay.
4. Optional: use `Prev/Next` to browse images in the same folder.

After `Detect`, the displayed image is the detector overlay returned by `detect/weigao_tray.py`.

## Config Load/Save Behavior

- Load priority:
  1. `config.yaml` (working config)
  2. `detect_weigao_tray.yaml` (defaults fallback, packaged app)
  3. `config/detect_weigao_tray.yaml` (defaults fallback, source repo compatibility)
- Save target after successful `Detect`:
  - `config.yaml`
- If params were not changed, no file write occurs.

## Project Layout

- `main.py`: thin application entrypoint
- `app/__init__.py`: app package export surface
- `app/controller.py`: app workflow controller
- `ui/control_panel.py`: right parameter panel
- `ui/image_view.py`: image canvas, overlays, anchors, nav hit areas
- `core/grid_utils.py`: ROI grid generation helpers
- `core/params.py`: YAML load/merge/save
- `core/detector.py`: adapter to synced detector
- `detect/`: synced detector package from VisionRuntime
- `config/`: synced detector config
- `data/`: sample images

## Docs

- `docs/app_design.md`: weigao-tray-tuner app design and runtime flow
- `docs/detect.md`: detector reference and compatibility constraints
- `docs/sync_contract.md`: sync contract with VisionRuntime
- `docs/ui_param_mapping.md`: UI label to detector key mapping
