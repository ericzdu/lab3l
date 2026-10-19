# Team setup and contribution guide — Emma development branch

This is a **work-in-progress simulation prototype**, not a finished robot
controller or a trained policy. The code is under `shadow-hand-teleop/` in
`ericzdu/lab3l`. Start here rather than assuming that downloading the repository
also installs Python, dependencies or robot models.

## 1. Get the development baseline and make your own branch

For a new checkout:

```bash
git clone --branch Emma https://github.com/ericzdu/lab3l.git
cd lab3l
git switch -c YourName
cd shadow-hand-teleop
```

Replace `YourName` with your actual name, for example `Kyle`. If that branch
already exists, use `git switch YourName` instead of recreating it. Keep each
person's work on their own branch; do not push directly to someone else's branch.

For an existing clean checkout, first fetch and inspect the development branch:

```bash
git fetch origin
git switch -c YourName origin/Emma
cd shadow-hand-teleop
```

Do not discard local changes to follow these commands. Commit or otherwise
preserve your work first. If your clone uses a different remote name, substitute
that name for `origin`. Opening a PR does not require merging it immediately;
agree on the target branch with the team.

## 2. Install a local environment

The primary tested platform is **macOS / Apple Silicon with Python 3.12**.
Use a Python 3.12 installation that includes `tkinter` (the separate controls
window). Linux commands are included below but have not been tested in this
release; the `.command` launchers are for macOS. Windows is not validated.
No NVIDIA GPU is required for this MuJoCo prototype; a graphical desktop and
working graphics driver are required for the viewer.

From `shadow-hand-teleop/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m tkinter
```

The last command should open a small Tk test window. Close it before continuing.
If `tkinter` is missing, install matching Tcl/Tk support for your Python build
using your Python installer or OS package manager. It is not installed by pip's
`requirements.txt`. Do not copy someone else's `.venv`; environments contain
machine-specific paths.

On a desktop Linux installation with Python/Tk available, the environment and
pip commands above are also the intended setup path. Use `python` rather than
`mjpython` for the viewer there. Remote headless servers cannot show a native
window without a separately configured display.

## 3. Download models and run checks

The project downloads Shadow Hand / Panda assets through `mujoco-menagerie`.
The first run needs internet. Download them once into the ignored local cache:

```bash
MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" .venv/bin/python -c "import mujoco_menagerie as m; m.prefetch(['shadow_hand', 'franka_emika_panda'])"
.venv/bin/python scripts/check_setup.py
.venv/bin/python scripts/view_dual_arms.py --wrist human --workspace --headless --seconds 3
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

Headless checks validate model construction and offline physics. They do not
prove that a headset is connected or that live grasping works. Models and mesh
licenses remain those of their respective authors; assets are not bundled in
this branch.

## 4. Open an offline scene (no Vision Pro needed)

On macOS, double-click `run_human_arms.command` in Finder, or run:

```bash
./run_human_arms.command
```

That opens dual simplified human-shaped arms, Shadow Hands, a table, a yellow
cup and a teal tray. Use the separate **Holos — Hand controls** window for
robot keys/buttons; click the MuJoCo window only for camera/display controls.
The same letters can trigger native MuJoCo shortcuts in the wrong window.

- `1` / `2`: choose left / right arm.
- `W` / `S`, `A` / `D`, `R` / `V`: forward/back, left/right, up/down.
- `I` / `K`, `J` / `L`, `U` / `P`: wrist orientation adjustments.
- `F`: both hands close gradually over about 0.9 seconds.
- `O`: both hands open gradually over about 1.3 seconds.
- `H`: reset arms **and cup**; this discards the current scene pose.
- `Space`: pause/resume; `B`: operator/overview camera.

On Linux (not validated here), the equivalent viewer command is:

```bash
.venv/bin/python scripts/view_dual_arms.py --wrist human --workspace
```

For only the original fixed hands, use `run_dual_shadow_hands.command`.
`run_dual_arms.command` opens the separate Panda-derived/spherical-wrist variant.
The variants have different joint layouts: do not interchange state files.
The legacy `run_vision_pro_shadow_hand.command` requires a separate checkout
of the upstream VisionProTeleop examples under `third_party/`; those examples
are not bundled here. Use `run_live_human_arms.command` or the included original
`run_live_dual_hands.command` for this branch's own live controllers.

## 5. Optional live Vision Pro control

1. Open Tracking Streamer on the headset, press Start, and keep it worn/active.
2. Prefer the current Local Network IP when both devices share a Wi-Fi/LAN.
   A current external room code is another option. Network isolation may still
   block a same-Wi-Fi connection; a room code requires the external service.
3. On macOS run `./run_live_human_arms.command` and enter that IP or room code.
4. Face forward, keep both hands visible, hold wrists comfortably about shoulder
   width apart, and click **Calibrate · C** in the controls window.
5. Confirm `arm FOLLOW`. `forearm ON` means optional forearm data is usable and
   calibrated; `forearm unavailable` falls back to wrist-only control. Missing
   optional forearm data is not itself a network failure.

The Linux equivalent (not validated here) is:

```bash
.venv/bin/python scripts/view_dual_arms.py --wrist human --workspace --address CURRENT_IP_OR_ROOM_CODE
```

After tracking loss, pause/resume or H reset, press C again. C recenters input;
it does not move the physical simulation base. `ICE closed` plus `WAIT / HOLD`
means no fresh hand data is reaching this controller; inspect the terminal
before trying to calibrate repeatedly. Do not share personal recordings or
connection codes in commits.

## 6. Current limitations / useful tasks to claim

- **Fixed shoulder bases and fixed 28 cm / 26 cm arm links.** Some targets are
  unreachable or clipped by the workspace/posture envelope. `REACH LIMIT`
  reports requested-target clipping, not collision. A movable shoulder/torso
  base has been discussed but **is not implemented in this branch**.
- **Approximate live finger retargeting.** Finger spread and full thumb opposition
  are missing. Offline keyboard grasping has been observed, but live capture
  is not a validated reliable grasp controller. BunnyVisionPro/dex-retargeting
  has **not** been integrated; evaluating it is a useful separate task.
- **Release still needs testing.** O now opens gradually, but hand/cup/handle
  contacts can still push or flick the cup. Slow opening is not a guarantee
  of a gentle release or a force controller.
- **Idealized arms.** Shoulders are fixed; elbow posture is inferred/guided,
  wrist freedom is virtual, and gravity compensation is idealized. This is
  not a hardware-faithful robot or an anatomical digital twin.
- **No live arm-demo recorder or trained policy.** The live arm launcher controls
  the scene but does not save demonstrations. Existing mock export / HDF5
  replay tools are retained; they must not be described as recorded live-arm
  demonstrations. See README for their supported schemas/models.
- **Platform/UI validation.** Offline checks do not validate another person's
  headset, network, graphics driver or Tk installation. Start with keyboard
  mode before debugging streaming.

Possible work areas: arm control and movable base; hand retargeting; household
scenes and task metrics; live recording/export and model metadata; setup and
reproducibility. Post which area you are taking so others can avoid duplication.

## 7. Where to change things

| Area | Files |
|---|---|
| Human arm geometry / IK / reach limits | `scripts/human_arms.py` |
| Input calibration / coordinate mapping | `scripts/arm_control.py`, `scripts/forearm_tracking.py` |
| Live packet bridge / approximate fingers | `scripts/teleop_shadow_hands.py`, `scripts/hand_retargeting.py` |
| Cup / tray / tabletop props | `scripts/workspace_scene.py` |
| Viewer loop / gradual opening | `scripts/view_dual_arms.py` |
| Separate controls window | `scripts/control_panel.py` |
| Existing export / playback | `scripts/collect_mock_demo.py`, `scripts/replay_data.py`, `scripts/replay_demo.py` |

Most model code uses metres, radians and kilograms. Keep body names and actuator
names unique when adding models. Preserve teammates' replay modules and the
original hand-only launchers when experimenting with another robot.

## 8. Share your changes

From the repository root, review `git status`, run the relevant tests, and stage
only the files you intended to change. Example for someone changing the scene:

```bash
git add shadow-hand-teleop/scripts/workspace_scene.py
git commit -m "Update household task scene"
git push -u origin YourName
```

Then open a PR, state what changed, list checks performed, and document remaining
issues. Keep `.venv`, `.cache`, downloaded models, recordings, personal file paths
and credentials out of Git. License notices and attribution are required when
reusing third-party code; neither BunnyVisionPro nor a full live recorder is
present merely because it is listed as a possible future task.
