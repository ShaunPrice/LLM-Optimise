# LLM-Optimise video visual system

`marketing/video/scripts/visuals.py` is a deterministic Pillow compositor. Import its `draw_scene(scene, t, duration)` function; `scene` is an entry from `narration-scenes.json`, while `t` and `duration` are seconds within that scene. It returns an RGB Pillow image at 1920 × 1080. Rendering does not invoke a model, network service or browser.

The bottom 140 pixels, from y=940 through y=1079, remain solid navy for the assembly script's voice-aligned subtitles. Scene transitions, narration, music, caption timing and video encoding belong to the separate assembly stage. Every scene has continuing motion: sparse moving particles, a gentle screenshot push, flowing diagram connectors, progress cues or an orbiting brand accent. Reveals finish within the opening second; there are no flashing transitions or invented dashboard animations.

## Direction

Deep navy `#0b1728`, cobalt `#2f62cb`, teal `#3cd3c5` and ivory form a restrained laboratory identity. Avenir Next supplies the typography on macOS; Helvetica, DejaVu Sans and Arial are fallbacks. The existing application icon is used unchanged for identity. The existing README header informed the palette and processor/path motif; it is not repeated as a flat slide.

UI scenes use a narrow editorial title column and a large, framed genuine screenshot. Screenshots are intentionally cropped to show the relevant workflow and exclude the persistent workspace footer. No metrics or interface controls are fabricated. The comparison scene magnifies the existing measured scatter plot. The training capture is recipe preparation, explicitly labelled as such; it is not presented as a recording of the completed training run.

The architecture separates GUI/CLI/MCP entry points, the shared Python workflow engine, optional Rust process supervision with llama.cpp, and independent Soup, cloud-provider and Docker branches. It does not place model kernels inside Rust. The MCP scene presents compatible client options and explicitly says remote HTTPS/OAuth requires gateway setup. It does not claim a live ChatGPT connection. The validation scene separates software CI, recorded hardware workflows and integration checks; it does not use a test count that can drift.

## Privacy redactions

The compositor preserves the source screenshot files and overlays narrow, opaque masks labelled **Local path omitted** on the reviewed private-path rows. The Develop capture masks its saved-evidence path at source rectangle `(518,493)-(1132,515)`. The Training capture masks its configuration path at `(540,65)-(1132,91)` and CLI path at `(542,143)-(1132,188)`. Coordinates use the 1185 × 782 source space and scale proportionally. These two scenes disclose **Actual application / local paths omitted** in their visual footer. The masks do not alter model output, test status, plotted values or performance measurements. The other crops omit workspace footers and the lower router executable path.

## Required assets

All asset paths are relative to the repository, resolved from this script's location:

- `docs/assets/app-icon.png`
- `marketing/video/narration-scenes.json`
- `marketing/video/shots/lab.png`
- `marketing/video/shots/results.png`
- `marketing/video/shots/workbench-result.png`
- `marketing/video/shots/routing.png`
- `marketing/video/shots/develop.png`
- `marketing/video/shots/training.png`
- `marketing/video/shots/chat.png`

`workbench.png` has a crop preset and can be selected for the exploration scene by changing its screenshot mapping. Crop presets use the 1185 × 782 source coordinate space and scale proportionally for higher-density captures. Review crops again whenever a screenshot is replaced or its scroll position changes. File modification time invalidates the in-process screenshot cache.

Normal `draw_scene` rendering raises `FileNotFoundError` for a missing screenshot. Only the explicit proof CLI permits a clearly labelled pending-capture rectangle. Production assembly must use normal rendering and assert the complete asset set.

## Proofs and validation

Run with a Python environment containing Pillow and NumPy:

```sh
python marketing/video/scripts/visuals.py --proof
python marketing/video/scripts/visuals.py --proof --scene 5
```

This renders midpoint `shots/proof-NN-name.png` frames and `shots/proof-contact-sheet.png`. Each proof remains silent and leaves the subtitle area empty. Twelve scene pairs were checked for RGB output, exact dimensions, a clean caption reserve and non-identical motion frames. Twenty-four frames took approximately 0.39 seconds in an initial local smoke run; this is compositor-only timing and excludes audio, captions, encoding and first-time font/asset costs in other environments. Ruff checks pass.
