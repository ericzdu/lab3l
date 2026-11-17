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
- **3D mesh import (experimental):** load a mesh exported from SAM 3D or other
  tools onto the table (`--scene-config`).
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
| `./run_photo_scene.command` | The scene with an imported 3D mesh |

Run the tests:

```sh
MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" .venv/bin/python -m pytest
```

## Status

A working prototype, not a validated controller or a trained policy. Known
limits: fixed shoulder positions limit reach; finger spread and thumb rotation
are not retargeted; success labels are heuristics not yet checked against human
labels. Details: [TEAM_SETUP.md §6](shadow-hand-teleop/TEAM_SETUP.md).
In progress: real 3D object models with accurate collision shapes (HOL-49).

## Repository layout

| Path | Contents |
|---|---|
| `shadow-hand-teleop/` | Simulation, teleoperation, recording, Studio and tests ([README](shadow-hand-teleop/README.md) for detailed notes) |
| `shadow-hand-teleop/docs/` | Specs for recording/labels, Studio, headset view, reachable workspace |
| `ui/` | Read-only research UI (React) |
| `.github/workflows/` | CI: pytest on pushes to `main` and on PRs |
