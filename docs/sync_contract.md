# Sync Contract with VisionRuntime

This document defines how weigao-tray-tuner stays aligned with the main project:

- Main project: `<VISIONRUNTIME_ROOT>`
- Tuning project: `<WEIGAO_TRAY_TUNER_ROOT>`

## Source of Truth

For detector behavior and default parameters, VisionRuntime is the only source
of truth.

## Files to Sync

Manually copy from VisionRuntime to weigao-tray-tuner:

- `config/detect_weigao_tray.yaml` -> `config/detect_weigao_tray.yaml`
- `detect/*` -> `detect/*`
- `utils/image_codec.py` -> `utils/image_codec.py`
- (optional) sample images -> `data/images/*`

Do not sync weigao-tray-tuner UI/controller files back into VisionRuntime.

## Expected Local Overrides

weigao-tray-tuner may diverge from VisionRuntime in:

- `main.py`, `app/controller.py`, `ui/control_panel.py`, `ui/image_view.py`, `core/slot_layout_utils.py`, `core/params.py`, `core/detector.py`
- docs in `docs/*`
- local working config file `config.yaml`

These are tuning-shell concerns, not detector source-of-truth concerns.

## Post-sync Validation Checklist

Run this checklist after each sync:

1. `python -m py_compile main.py app/controller.py app/__init__.py ui/control_panel.py ui/image_view.py core/detector.py core/params.py core/slot_layout_utils.py`
2. Launch app and open one sample image.
3. Click `Detect` once and verify overlay appears.
4. Toggle debug switches and verify expected overlay visibility.
5. Confirm no key mismatch errors when editing each parameter section.

## Drift Handling Rules

When detector keys change upstream:

1. Update `ui/control_panel.py` mapping and labels.
2. Update `docs/ui_param_mapping.md`.
3. If stage semantics changed, update `docs/detect.md` summary section.

When new upstream detector keys are added:

- Either expose in UI, or list under "Hidden params" in `docs/ui_param_mapping.md`.
