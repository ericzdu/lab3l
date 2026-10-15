# Holos Vision Pro Teleoperation Proof of Concept

This project is the local simulation workspace for testing this pipeline:

```text
Apple Vision Pro hand tracking
        -> VisionProTeleop / avp_stream
        -> MuJoCo Shadow Hand
        -> robot-state and action demonstrations
        -> later policy training
```

## Current status

- MuJoCo 3.15 is installed in the local `.venv`.
- MuJoCo Menagerie and the Shadow Hand model are installed and cached locally.
- `avp_stream` is installed for the later Vision Pro connection.
- The official VisionProTeleop examples are available under `third_party/VisionProTeleop`.
- A simulation smoke test and a local Shadow Hand viewer are ready.

## Open the Shadow Hand simulation

Double-click `run_shadow_hand.command` in Finder. If macOS asks whether to open it,
choose **Open**.

The default launcher opens the left hand. To open the right-hand model instead,
double-click `run_right_shadow_hand.command`. Click inside either MuJoCo viewer,
then press **F** to make a fist or **O** to open the hand.

To show both hands in the same MuJoCo window, run
`run_dual_shadow_hands.command`. Its keyboard controls are:

- **1**: toggle the left hand between open and fist
- **2**: toggle the right hand between open and fist
- **F**: make both hands into fists
- **O**: open both hands

The same action can be run from Terminal:

```bash
./run_shadow_hand.command
```

## Live two-hand finger control (first prototype)

Run `run_live_dual_hands.command` in Finder and enter the CURRENT Tracking Streamer
IP or remote room code. Keep the headset and app active. A MuJoCo window on the
Mac shows both fixed Shadow Hands; this launcher does not stream the scene into
the headset. SPACE pauses/resumes; closing the window disconnects. No recordings,
camera input or microphone input are enabled. Remote mode uses third-party
signaling and, when needed, an encrypted TURN relay.

This first retargeter maps individual finger bend angles, clips joint targets,
and rate-limits commands. It does NOT reproduce the complete human hand pose:
spread, thumb opposition, wrist motion, and arm reaching are not implemented.
No fresh packets for 0.5 seconds freezes the full simulation if both hands are
stale; a single stale hand holds its last actuator target. The SDK does not expose
per-hand tracking validity in this interface, so a frozen but repeatedly sent
skeleton cannot always be distinguished from a deliberately still hand.
This is a connection/control prototype, not a calibrated grasping controller or
a trained policy. Two robot arms have not been added yet.

Offline checks: `.venv/bin/python scripts/test_hand_retargeting.py`.

## Mock teleoperation/data-collection demo

Before the Vision Pro is available, double-click `run_mock_demo.command`. The
simulated hand performs an open-close motion and saves a demonstration to:

```text
data/mock_grasp_demo.npz
data/mock_grasp_demo.json
```

The dataset contains timestamps, robot positions and velocities, commanded
actions, the object's pose, and the number of active contacts. This is a pipeline
test rather than a successful-task label; the outcome remains `not_evaluated`.

## Verify the local installation

```bash
MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" \
  .venv/bin/python scripts/check_setup.py
```

## Connect Apple Vision Pro later

1. Install and open **Tracking Streamer** on Apple Vision Pro.
2. Put the Vision Pro and this Mac on the same Wi-Fi.
3. Press **Start** in Tracking Streamer and note the displayed IP address.
4. Run:

```bash
.venv/bin/python scripts/check_vision_pro.py VISION_PRO_IP
```

Replace `VISION_PRO_IP` with the address shown in the headset. A room code can be
used instead when using VisionProTeleop's external-network mode.

After the basic stream works, the official Shadow Hand controller is located at:

```text
third_party/VisionProTeleop/examples/12_diffik_shadow_hand.py
```

For the first live test, you can instead double-click
`run_vision_pro_shadow_hand.command`. It asks for the IP address or room code and
then launches the official VisionProTeleop Shadow Hand controller.

## Planned milestones

1. Confirm live Vision Pro hand tracking on the Mac.
2. Replace the mock input with the existing VisionProTeleop Shadow Hand controller.
3. Record one real-time grasping demonstration with a fixed Shadow Hand.
4. Define one household manipulation task and evaluation metric.
5. Add a simulated robotic arm only after fixed-hand control is reliable.

## HDF5 recording and replay

Install the project requirements (including `h5py`) into `.venv` first.
To save a mock demonstration as HDF5:

```bash
.venv/bin/python scripts/collect_mock_demo.py --output data/mock_grasp_demo.h5
```

Replay it in the MuJoCo viewer on macOS:

```bash
.venv/bin/mjpython scripts/replay_demo.py data/mock_grasp_demo.h5
```

Use `--speed 0.5` for half speed or `--speed 2` for double speed. Close the
viewer to stop. For validation without a window:

```bash
.venv/bin/python scripts/replay_demo.py data/mock_grasp_demo.h5 --headless
```

The player also accepts `.hdf5` and existing `.npz` demonstrations. Playback
restores recorded states with their original sample timing; it does not rerun
physics from actions. Files must match the default single left Shadow Hand model.

HDF5 files use root datasets `timestamp` (N,), `qpos` (N, nq), `qvel` (N, nv),
and `action` (N, nu). Times are seconds, finite, nonnegative, and strictly
increasing. States and actions must be finite numeric arrays. Mock exports also
include `object_pose` (N, 7; xyz then wxyz quaternion) and `contact_count` (N,).
The object's state is already included in `qpos`; those extra datasets are for
analysis. Exported HDF5 files include `schema_version=1` and JSON `metadata`
attributes, plus the same JSON sidecar as NPZ exports. Other HDF5 layouts need
conversion to these root dataset names and model dimensions before playback.

Offline replay checks: `.venv/bin/python scripts/test_replay_data.py`.

### HolosCapture exports

The player automatically recognizes HolosCapture HDF5 exports. For example:

```bash
.venv/bin/mjpython scripts/replay_demo.py "$HOME/Downloads/2026-03-13T17-38-41-D834FA49-D935-4FAA-A4BA-C816E3C162AC.h5"
```

Use `--demo demo_1` to select another demonstration (default: `demo_0`).
The adapter reads `data/<demo>/time` and
`obs/hand_{left,right}_joints_world`, using each dataset's `joint_names`
attribute or the root `joint` labels to reorder the skeleton. Parent-local
joint positions are not used as world coordinates. NaN or degenerate hand
frames hold that hand's last target; a hand starts neutral until valid tracking.

Holos playback first retargets the capture through the existing approximate
finger-flexion controller and simulates both fixed Shadow Hands, then plays
those generated states at the recorded times. Preparation may take a few seconds.
This is a reconstruction from human tracking, not recorded robot state. Wrist
and arm motion, captured objects, cameras, and audio are not replayed. The
capture's `actions` describe poses rather than Shadow actuator commands and
are not sent directly to the simulated hands. `--headless` performs the same
retargeting and simulation checks without opening a window.

Holos captures now open in a side-by-side comparison by default: **recorded
tracking on the left**, **moving Shadow reconstruction on the right**. The
skeleton preserves recorded world-space joint positions at metric scale. Both
panels share a fixed translation and Y-up to Z-up rotation; a constant horizontal
offset separates them. Blue is the left tracked hand, orange is the right.

Robot palm origins follow the recorded wrists. Wrist quaternions drive rotation,
with a fixed anatomical alignment estimated from the first usable pose for each
hand. Robot proportions and the approximate finger retargeter still differ from
the human skeleton. Missing skeleton frames disappear; robot wrists and finger
targets hold their previous pose. A robot stays hidden until its first valid wrist
pose. This display moves the robot mounts kinematically; it does not simulate a
robot arm or interactions with captured objects. This comparison supersedes the
fixed-wrist description above. NPZ and flat robot-state HDF5 playback are unchanged.

### Rising smoke interaction

The robot comparison panel includes smoke emitters beneath the recorded hand
workspace. Buoyant particles rise, curl, and deflect along MuJoCo hand surfaces;
they diffuse sideways when blocked by a palm or finger. Puffs use their full visible radius for solid clearance. Batched distance checks against conservative oriented bounds resolve overlaps
even when a moving hand sweeps into smoke; puffs that cannot fit between solids
are removed. These bounds enclose collision shapes and visible meshes; smoke
may deflect slightly before reaching the detailed surface. Press **S** to hide/show the plume. The original
tracking panel stays clear for comparison.

Smoke emits 24 puffs per second with a limit of 96 particles. Display updates
are capped at 30 Hz and follow elapsed recording time, skipping overdue frames
instead of accumulating playback delay. Hiding smoke with **S** also stops its
collision processing.
This is a bounded, lightweight particle approximation, not a volumetric fluid
solver. Smoke does not exert forces on the prescribed replay motion. Emitters
stay fixed beneath the capture workspace. Smoke timing follows recording time,
including when playback speed changes. Smoke collision checks are also exercised
by `--headless`; the final status reports surface interactions.

Offline smoke checks: `.venv/bin/python scripts/test_smoke_plume.py`.
