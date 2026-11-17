hello world
Holos Team 1 Project

Description: This project explores using Apple Vision Pro as a real-time teleoperation interface for robotic manipulation and demonstration collection. Building on the existing VisionProTeleop framework, we aim to map Vision Pro hand-tracking data to a simulated Shadow Hand in MuJoCo, enabling users to control robotic movements while recording robot states, actions, object poses, and task outcomes. We plan to extend the system with robotic arms for object manipulation and collect demonstrations of household tasks to support robot policy training.

## Emma development baseline

Start with [Team setup and contribution instructions](shadow-hand-teleop/TEAM_SETUP.md).
The `Emma` branch includes experimental arm-equipped scenes and known limitations;
it is not a release-quality controller. Create your own named branch from this
baseline, coordinate an area of work, and contribute through a PR. Existing
mock export and HDF5/Holos replay tools are preserved. The live arm viewer does
not yet save robot demonstrations; movable shoulder bases and BunnyVisionPro
integration remain open work.

## Importing a real 3D object into the simulation

Turn a photo of a real object into a simulated prop. Run the commands from
`shadow-hand-teleop/`. Full details: [ADDING_OBJECTS.md](shadow-hand-teleop/docs/ADDING_OBJECTS.md).

1. **Install the mesh tools (once):**
   ```sh
   .venv/bin/python -m pip install -r requirements-photo-scene.txt
   ```
2. **Photograph the object.** One object per photo, plain background, even
   light, a three-quarter view from slightly above.
3. **Make a mesh.** Open Meta's [SAM 3D demo](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d),
   upload the photo, click the object, generate it and download it as **GLB**
   (not a Gaussian splat / PLY point cloud).
4. **Measure the real object:** height in metres (ruler) and mass in kg
   (kitchen scale). For flat round things (plates) measure the diameter and
   use `--width` instead of `--height`.
5. **Import it** into a new folder under `assets/objects/`:
   ```sh
   .venv/bin/python scripts/prepare_object.py ~/Downloads/mug.glb assets/objects/mug \
     --id mug --up-axis y --height 0.095 --mass 0.32 --measured height mass \
     --source-kind sam3d_export --keep-source
   ```
   Useful options: `--footprint round` for plates/bowls (scored as round
   targets), `--collision box` for box-shaped objects, `--condim 6` for round
   objects that should not roll away, `--align-footprint` if a box came out
   rotated.
6. **Look at it on the table:**
   ```sh
   zsh ./run_photo_scene.command assets/objects/mug
   ```
   If it lies on its side, delete the folder and retry with another
   `--up-axis` (`x` or `z`). Toggle geom group 3 in the viewer to see the
   collision shape.
7. **Use it in a task (optional):** add a prop to `PROPS` in
   `scripts/workspace_scene.py`, e.g.
   `"real_mug": {"type": "asset", "asset": "mug", "xy": (.24, .04)}`, then run
   `./run_human_arms.command --props real_mug`, or point a task in `TASKS` at it.
8. **Check and commit:** run
   `MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" .venv/bin/python -m pytest scripts/test_object_assets.py`
   and commit the new `assets/objects/<id>/` folder. Only commit meshes you
   have the right to share; add the license with `--license` / `--attribution`
   for third-party models.

The apple, block, big block and plate used by the `_real` tasks
(`./run_human_arms.command --task apple_to_plate_real`) are currently YCB scans
used as placeholders. To replace one, import your mesh as above and change
that prop's `asset` in `PROPS`.
