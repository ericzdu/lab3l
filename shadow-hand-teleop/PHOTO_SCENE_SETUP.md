# Photo → 3D mesh → offline MuJoCo: feasibility prototype

This adapter connects an **existing triangle-mesh export** to the local human-arm
workspace. It is not a SAM 3D model implementation and does not train a policy.
The original keyboard and Vision Pro launchers remain available and unchanged.

## Team-branch integration

This experimental branch starts from `Emma` at `08edefe` and retains the existing
Studio/recording/headset work. It adds an optional `--scene-config` route rather
than replacing the arm viewer or workspace. The team's current tabletop is at
**Z = 0.60 m**; the synthetic fixture and converter defaults use that height.

Episode recording is deliberately disabled **only for `--scene-config` runs**:
the current episode format/replay does not package these imported mesh assets.
Normal launches retain their existing recorder. This experiment does not claim
Studio integration, imported-scene replay, task labels or training-ready data.

The generated coffee-machine mesh, its texture and the source rendering are not
committed. A fresh clone can use the hand-authored fixture after installing the
optional dependencies. `run_coffee_scene.command` is a convenience entry point
only after placing a locally prepared bundle at the documented path.

Integration verification on 2026-11-13: the team-branch suite ran **168 tests**
successfully, with **3 skipped** (165 executed/passed); the added mesh-adapter
suite independently passed all **13 tests**. These are local checks, not a claim
that GitHub CI, live headset operation, or imported-scene recording was tested.
The user's supplied MuJoCo screenshot confirms the original standalone coffee
scene was visible in their native viewer; the team-branch default instead ships
the synthetic carton and uses the updated 0.60 m table.

## What works locally

- Import a local GLB, GLTF, OBJ, STL, or triangle-mesh PLY.
- Bake scene-node transforms; retain separate mesh parts.
- Set an explicit source up-axis and a known/assumed height. Uniform scaling
  preserves aspect ratio. The prepared mesh becomes Z-up, in metres, with its
  bottom centre at the body origin.
- Preserve base-color UV textures when available (up to 2048 px); use one mean
  color for vertex/face-colored parts. Full PBR/transparent materials are not
  reproduced.
- Add movable or fixed objects with explicit mass, friction and collision.
- Run without a Vision Pro; the same arm controls, table, original cup and tray
  remain. H resets imported objects as well as the original workspace.
- Keep provenance and manual assumptions in a portable JSON bundle.

The included coffee-carton mesh is **hand-authored test geometry**, not extracted
from the supplied coffee-counter rendering, not a real scan and not SAM output.

## Start the offline test

From this project folder, using its existing Python 3.12 environment:

```sh
./.venv/bin/python -m pip install -r requirements-photo-scene.txt
zsh ./run_photo_scene.command
```

Use the separate **Holos — Hand controls** window for robot keys:

- 1/2: select a hand; W/S, A/D, R/V: move the wrist target.
- I/K, J/L, U/P: rotate; F/O: gradually close/open both hands.
- H: reset; B: switch camera; Space: pause.
- Use the MuJoCo window for the camera, not these robot keys.

Imported visuals are in geom group 2; collision proxies in group 3, hidden by
default in this launcher. Enable group 3 in the viewer to inspect the collider.
The default scene adds an orange test carton on the existing table.
Reach limits are unchanged: this task does not extend the arms or move shoulders.

## Generate a real mesh first (separate upstream step)

Official sources:

- [SAM 3D Objects repository](https://github.com/facebookresearch/sam-3d-objects)
- [Official setup requirements](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Official image-to-3D demo](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d)

The published local inference setup requires Linux and an NVIDIA GPU with at
least 32 GB VRAM. This Mac adapter does **not** install that inference stack.
The online demo is a separate service with its own login, terms and availability.

For the supplied coffee-counter image, start with **one clearly visible object**
(for example, a coffee machine or grinder), select its mask, generate the model,
and download a triangle mesh, preferably GLB. A Gaussian-splat PLY is not a
triangle mesh and is deliberately rejected. Do not import the whole counter as
one dynamic object and claim all its small parts are individually manipulable.

This input is an architectural rendering with no known scale or calibrated
camera. Hidden surfaces are guesses; image generation does not establish mass,
friction, material strength, articulation, or functionality of a coffee machine.
Check source-image/asset rights before redistributing them. The input image is
not bundled into this repository by this prototype.

## Prepare a downloaded model

Example: use a **new** output folder (never overwrite an existing bundle):

```sh
./.venv/bin/python scripts/prepare_photo_object.py /absolute/path/object.glb \
  local_scene_assets/coffee_machine_v1 \
  --id coffee_machine --height 0.30 --up-axis y \
  --position 0.78 0.20 0.60 --mass 2.0 --fixed \
  --collision box --provenance sam3d_export

zsh ./run_photo_scene.command local_scene_assets/coffee_machine_v1/scene.json
```

Height 0.30 m and mass 2 kg above are **illustrative assumptions**, not measured
from the supplied image. Use the actual height where known and verify up-axis
visually. `--fixed` suits a countertop appliance; omit it for a movable prop.
`--provenance sam3d_export` is a declaration, not independent inference proof.
The bundle records the source filename/hash and `inference_run_verified: false`.

Each bundle contains `scene.json`, normalized OBJ parts, and optional PNG
textures. Paths inside the JSON are relative and cannot escape the bundle.
To assemble several objects, copy their asset subfolders under one scene folder
and combine their object entries into `objects`, with unique IDs and updated
relative `parts[].mesh`/`texture` paths. No server connection occurs on load.

## Collision and physics: important distinction

- `box`: one bounding box. Fast, but cannot model a cup's opening or a handle.
- `convex`: one convex hull of all visual parts. More detailed exterior, but
  **fills holes, cavities, gaps between parts, and handle openings**.
- `boxes`: manually fitted boxes, e.g. separate walls/base to leave an opening.
  Edit the JSON with an explicit list, in the object's local metre coordinates:

```json
"collision": {
  "mode": "boxes",
  "boxes": [
    {"position_m": [0, 0, 0.005], "half_size_m": [0.04, 0.04, 0.005]},
    {"position_m": [0.035, 0, 0.06], "half_size_m": [0.005, 0.04, 0.05]}
  ]
}
```

That two-box example is NOT a complete cup. Fit all required walls yourself.
MuJoCo computes inertia from the collision geometry, distributing the specified
mass by proxy volume; it does not recover real inertia from the photograph.
The visual meshes have zero mass and no contacts. There is no invisible weld or
grasp assist in this adapter. Successful import is not proof of stable grasping.

## Verify

```sh
./.venv/bin/python -m unittest discover -s scripts -p test_photo_scene.py -v
zsh ./run_photo_scene.command --headless --seconds 3
```

Tests cover formats, scene transforms, metric normalization, texture/UV loading,
input rejection, collision proxies, gravity, tabletop support, reset, and existing
arm/cup/tray integration. They do not test SAM inference, headset operation,
realistic contact parameters, policy training, or a headset frame-time budget.

## Claims supported at this stage

“We implemented and tested an offline mesh-to-simulation adapter, as an initial
step in investigating image-derived 3D assets for robotic simulation.”

Do not claim “photo-to-simulation is complete” until an actual SAM reconstruction
has been generated, imported, inspected and evaluated. Do not mark the broader
HOL-49 requirements complete based on this single synthetic test asset.

## Original standalone run: supplied coffee-counter image (2026-11-13)

The official web demo generated one coffee-machine object from a point-selected
mask of the user's reference rendering, after the user completed login/upload.
The downloaded `object_0.glb` has **14,995 vertices / 22,122 triangles**, one
textured part, and a size of 1,585,216 bytes. Its SHA-256 is
`a4de86d16120c6d1f5462e4e0d60799c42c3f6a47da5099ad3be65764bd0cac4`.

The prepared local bundle is `local_scene_assets/coffee_machine_v1/scene.json`.
This directory is gitignored; the generated asset is NOT automatically shared
with teammates. On this Mac, open it with:

```sh
zsh ./run_coffee_scene.command
```

It uses a manually assumed **0.30 m height**, Y-up conversion, -90-degree yaw,
position `[0.78, 0.20, 0.505]`, a fixed body, and one box collision proxy.
That measurement/run used the earlier standalone Z = 0.505 m table. When
preparing a new bundle for this team branch, set its bottom Z to **0.60 m**.
The resulting unrotated size is about 0.621 × 0.457 × 0.300 m. The 2 kg mass and
friction coefficients are assumptions; the fixed appliance is not a grasp task.
The existing cup/tray can still be used for keyboard manipulation experiments.
The bundle's provenance was updated for this observed web run; the generic
converter still defaults to unverified provenance for arbitrary input files.

The coffee launcher passed a 3.00 s headless run with finite state, 50 original
actuators and reported 0.00 mm wrist errors at rest. A separate local diagnostic
found no appliance/robot contact at home, approximately 0.574 s setup/compile
time, and 0.668 ms average for one offline control update plus five 2 ms physics
steps over 100 iterations. These are a single-machine CPU smoke measurement,
not an interactive/rendering/streaming latency or headset frame-rate guarantee.

Verified: official 3D preview, local GLB/texture parsing, MuJoCo compilation,
and headless physics integration. The complete local test suite passed 57 tests
(13 added for this adapter). Screenshot rendering from the automation terminal
failed with `invalid CoreGraphics connection`, so the simulation viewer's
appearance, reconstructed orientation, and interactive performance still need
visual confirmation in a normal Mac session. No headset, policy, robot hardware,
or accurate real-world contact parameters were validated by this work.
