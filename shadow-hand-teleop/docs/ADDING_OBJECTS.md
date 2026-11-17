# Adding real 3D objects to the simulation

Real objects live in `assets/objects/<id>/`, one folder per object, made by
`scripts/prepare_object.py` from any GLB/GLTF/OBJ/STL (a SAM 3D export, a scan
or a CAD model). The sim loads them with MuJoCo alone; `trimesh` and CoACD are
needed only to prepare a folder:

```sh
.venv/bin/python -m pip install -r requirements-photo-scene.txt
```

## What is in an object folder

| File | Purpose |
|---|---|
| `object.json` | Size, mass, friction, footprint, default placement, source and license |
| `visual_<i>.obj` (+ `.png`) | Look only: no contacts, no mass (geom group 2) |
| `collision/hull_*.obj` | Convex pieces that carry all contacts and mass (group 3, hidden by default) |
| `source/<file>` | The original export, with `--keep-source` (keep SAM 3D GLBs) |

Meshes are metres, +Z up, with the object's bottom centre at the origin. In
the sim the body origin is the object's bounding-box centre, so it rests at
`table_top + height / 2`, like the primitive props. The object's mass is split
over its collision pieces by volume, and MuJoCo derives the inertia from them.
Toggle geom group 3 in the viewer to see the collision model.

## 1. Make a mesh with SAM 3D

1. **Photograph the object.** One object per photo, a plain background, even
   light, a three-quarter view from slightly above so the top and one side show.
   Use the actual objects the team handles.
2. Open the [SAM 3D Objects demo](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d),
   log in, upload the photo and click the object to select its mask.
3. Generate the 3D model and download it as **GLB** (a triangle mesh). A
   Gaussian-splat or point-cloud PLY is rejected: it has no surface to collide with.

SAM 3D does not know the object's real size, mass or friction, and it guesses
the hidden back side. The local SAM 3D inference setup needs Linux and an
NVIDIA GPU with 32 GB+ VRAM; the web demo avoids that. Check you have the
right to commit a photo-derived mesh before adding it to the repository.

## 2. Measure the real object

| Value | How | Flag |
|---|---|---|
| Height (mm) | Ruler or calipers, standing as it rests on the table | `--height 0.072` (metres) |
| Width (mm) | Largest horizontal size, e.g. a plate's diameter; a check on the mesh's proportions | or `--width 0.14` |
| Mass (g) | Kitchen scale | `--mass 0.068` (kg) |
| Friction | Leave the default (1.0) unless it is clearly slippery (glazed, ~0.5) or grippy (rubber, ~1.5) | `--friction 0.5 0.005 0.001` |

Give exactly one of `--height`, `--width` or `--scale` (use `--scale 1` for a
mesh already in metres, e.g. a scan). Scaling is uniform: proportions come
from the mesh. List what you measured with `--measured height mass`; anything
not listed is recorded as `assumed` in `object.json`.

## 3. Prepare the folder

```sh
.venv/bin/python scripts/prepare_object.py ~/Downloads/apple.glb assets/objects/apple \
  --id apple --up-axis y --height 0.072 --mass 0.068 --measured height mass \
  --footprint round --condim 6 --source-kind sam3d_export --keep-source
zsh ./run_photo_scene.command assets/objects/apple     # look at it on the table
```

- `--up-axis`: GLB is usually `y`. If the object lies on its side, try another.
- `--align-footprint`: squares the top view to the axes first. Use it for
  boxes that came out rotated (the YCB block was scanned at 45 degrees, so its
  "width" was its diagonal). `--rotate-z DEG` turns it by hand.
- `--footprint round|box`: the top's shape when the object is a task *target*
  (round for plates and bowls, box for blocks).
- `--condim 6` adds rolling friction (round things, so they do not roll away);
  the default 4 adds torsional friction.
- An existing folder is never overwritten: prepare into a new folder.

### Choosing the collision model

| `--collision` | Use for | Notes |
|---|---|---|
| `coacd` (default) | Concave or irregular objects: plates, bowls, mugs, handles | Convex decomposition. `--coacd-threshold` lower = closer fit, more pieces (0.05 default; the plate uses 0.01 with `--max-hulls 32`). |
| `convex` | Nearly convex objects | One hull; fills cavities and gaps. |
| `box` | Box-shaped objects (blocks, cartons) | A MuJoCo box primitive: exact, stable box-on-box contacts. |

Coarser is not always worse. The apple uses CoACD at threshold 0.2 (2 pieces):
at 0.05 it got 11 lumpy pieces and slipped out of the scripted grasp. The
blocks use `box`: the scan's rounded edges made the palm-down grasp fail. The
plate needed the fine setting: with 16 pieces a block set on it rocked between
pieces and never settled. Check yours with the tests below.

## 4. Use it in a task

Add a prop in `scripts/workspace_scene.py`:

```python
"real_apple": {"type": "asset", "asset": "apple", "xy": (.24, .04), "replaces": "apple"},
```

`replaces` names a primitive prop at the same spot; the two are never put in
one scene. Add or point a task at it in `TASKS` (e.g. `apple_to_plate_real`).
Run it:

```sh
./run_human_arms.command --task apple_to_plate_real
.venv/bin/python scripts/scripted_prop_tasks.py --task apple_to_plate_real
```

Labels read the object's height and footprint from `object.json`, so success
scoring needs no code change. Episodes store each real prop's object id,
sha256 of its folder, source kind and license under `metadata.scene.assets`;
replay warns if the files changed since recording.

To swap a placeholder for a SAM 3D mesh, prepare the new folder and change the
prop's `asset`. Old episodes still replay (same joints) with a warning that
the model changed.

## 5. Check it

```sh
MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" .venv/bin/python -m pytest \
  scripts/test_object_assets.py scripts/test_workspace_props.py scripts/test_prop_tasks.py
```

These check that every committed object is valid and credited; each real
prop rests on the table, resets and is reachable; visuals have no mass or
contacts; every scripted task (including `_real`) labels as SUCCESS; and with
all real models in the scene, a control update (five physics steps) stays
within the 16 ms headset frame budget (95th percentile) and the scene loads
in under 2 s.

## Imported scenes (`--scene-config`)

For objects outside the task system (e.g. a countertop appliance), pass one
object folder (placed at its `placement` in `object.json`) or a scene file:

```json
{"version": 2, "objects": [
  {"id": "coffee_machine", "asset": "coffee_machine", "position_m": [0.78, 0.20, 0.60],
   "yaw_degrees": -90, "fixed": true}
]}
```

`asset` is an id under `assets/objects/` or a folder relative to the scene
file (it cannot point outside it). `position_m` is the bottom centre. Imported
bodies are named `photo_<id>`; episodes record them like props and store the
scene's path and sha256, so replay rebuilds them. Version-1 bundles from the
original prototype (PHOTO_SCENE_SETUP.md) still load.

## Current objects

| Folder | Used by | Source | Notes |
|---|---|---|---|
| `ycb_apple` | `real_apple` | YCB 013_apple, true size | Placeholder; 150 g kept from the primitive apple |
| `ycb_plate_14cm` | `real_plate` | YCB 029_plate, 26 cm → 14 cm | Placeholder; scaled to stay in reach |
| `ycb_block_5cm` | `real_block` | YCB 070-b wood block, ~25 mm → 50 mm cube | Placeholder; box collision |
| `ycb_block_7cm` | `real_big_block` | Same scan → 70 mm cube | Placeholder; box collision |
| `test_coffee_carton` | `assets/photo_scene_test/scene.json` | Hand-authored | Importer test geometry |

Masses of the placeholders are the primitive props' values, not the YCB
objects' real masses. YCB credit: [assets/objects/ATTRIBUTION.md](../assets/objects/ATTRIBUTION.md).
