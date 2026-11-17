# lab3l — Holos Team 1

Apple Vision Pro as a real-time teleoperation interface for robot manipulation
in simulation. Hand tracking drives simulated Shadow Hands on human-shaped arms
in MuJoCo, and every attempt is recorded as a labeled demonstration for later
robot policy training.

## What's built

- **Teleoperation:** live Vision Pro hand tracking retargeted to two Shadow
  Hands (finger bend, estimated thumb opposition) on human-shaped arms with
  wrist position and orientation; keyboard control works without a headset.
- **Tabletop tasks:** a table with a cup and tray, plus optional props (block,
  bottle, plate, apple, big block) and scored tasks: `cup_to_tray`,
  `apple_to_plate`, `block_to_plate`, `block_on_big_block`, `set_the_table`.
- **Recording:** episodes saved as HDF5 with robot state, inputs, object poses,
  contacts and video, automatically labeled with phases and success/failure;
  replay in the MuJoCo viewer. Scripted demos solve each task end to end.
- **lab3l Studio:** a local browser page to run a session: connect, calibrate,
  record and review each result with its video and phase timeline.
- **Headset view:** stream the rendered simulation back into the Vision Pro.
- **Capture replay:** HolosCapture recordings replayed side by side with the
  retargeted robot hands.
- **Real 3D objects:** import a mesh (e.g. from a photo via SAM 3D) as an
  object with an accurate collision shape and real size and mass; `_real`
  copies of the prop tasks use real models (YCB scans as placeholders).
- **Research UI (`ui/`):** a read-only browser view of recorded datasets.

## Quick start

Setup, platforms and the full guide: [shadow-hand-teleop/TEAM_SETUP.md](shadow-hand-teleop/TEAM_SETUP.md).
From `shadow-hand-teleop/` (macOS, Python 3.12):

| Command | What it opens |
|---|---|
| `./run_human_arms.command` | Arms, hands and table; keyboard control, no headset |
| `./run_human_arms.command --task set_the_table` | The same with a scored task and its props |
| `./run_live_human_arms.command` | Live Vision Pro control (asks for the headset's IP) |
| `./run_studio.command` | lab3l Studio at http://127.0.0.1:8765 |
| `./run_human_arms.command --task apple_to_plate_real` | A task with real 3D models |
| `./run_photo_scene.command <object folder>` | The scene with an imported 3D object |

Run the tests:

```sh
MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" .venv/bin/python -m pytest
```

## Importing a real 3D object

From `shadow-hand-teleop/`, once: `.venv/bin/python -m pip install -r requirements-photo-scene.txt`.

1. Photograph the object alone on a plain background, three-quarter view from above.
2. In Meta's [SAM 3D demo](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d),
   upload it, click the object, generate, and download the **GLB**.
3. Measure its height (or a plate's diameter) and weigh it.
4. Import and view it:
   ```sh
   .venv/bin/python scripts/prepare_object.py ~/Downloads/mug.glb assets/objects/mug \
     --id mug --up-axis y --height 0.095 --mass 0.32 --measured height mass \
     --source-kind sam3d_export --keep-source
   zsh ./run_photo_scene.command assets/objects/mug
   ```
   Lying on its side? Delete the folder and retry with `--up-axis z` or `x`.
5. To use it in tasks, add it to `PROPS` in `scripts/workspace_scene.py`
   (`"real_mug": {"type": "asset", "asset": "mug", "xy": (.24, .04)}`), then
   `./run_human_arms.command --props real_mug`.

Options (round targets, box collision, rolling, rotated boxes) and how to
choose the collision model: [ADDING_OBJECTS.md](shadow-hand-teleop/docs/ADDING_OBJECTS.md).

## Status

A working prototype, not a validated controller or a trained policy. Known
limits: fixed shoulder positions limit reach; finger spread and thumb rotation
are not retargeted; success labels are heuristics not yet checked against human
labels. Details: [TEAM_SETUP.md §6](shadow-hand-teleop/TEAM_SETUP.md).
The real-object placeholders still need replacing with SAM 3D meshes of the team's own objects.

## Repository layout

| Path | Contents |
|---|---|
| `shadow-hand-teleop/` | Simulation, teleoperation, recording, Studio and tests ([README](shadow-hand-teleop/README.md) for detailed notes) |
| `shadow-hand-teleop/docs/` | Specs for recording/labels, Studio, headset view, reachable workspace; adding objects |
| `shadow-hand-teleop/assets/objects/` | Real 3D objects (one folder each) |
| `ui/` | Read-only research UI (React) |
| `.github/workflows/` | CI: pytest on pushes to `main` and on PRs |
