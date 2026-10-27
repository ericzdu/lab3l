# Holos Vision Pro Teleoperation Proof of Concept

> Development branch: start with [TEAM_SETUP.md](TEAM_SETUP.md) for installation, supported platforms, known limitations and contributing. Historical development notes follow.

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
and rate-limits commands. Thumb opposition (THJ4) follows how far your thumb
metacarpal leaves the palm plane (about 10° → 0, 45° → full), and rises toward
the keyboard fist's value as your fingers close. It does NOT reproduce the
complete human hand pose: spread, thumb rotation (THJ5), wrist motion, and arm
reaching are not implemented.
No fresh packets for 0.5 seconds freezes the full simulation if both hands are
stale; a single stale hand holds its last actuator target. The SDK does not expose
per-hand tracking validity in this interface, so a frozen but repeatedly sent
skeleton cannot always be distinguished from a deliberately still hand.
This is a connection/control prototype, not a calibrated grasping controller or
a trained policy. This original launcher keeps both wrists fixed; the optional
arm-equipped version below has a separate launcher.

Offline checks: `.venv/bin/python scripts/test_hand_retargeting.py`.

## Two Panda arms with Shadow Hands

Double-click `run_dual_arms.command` for an offline keyboard-controlled viewer.
The default now uses simplified **spherical wrists**, described below. The
previous Panda assembly remains available with `./run_dual_arms.command --wrist panda`.

### Simplified spherical wrists (default)

The large terminal Panda motor housings, joints 6/7, and the two Shadow wrist
hinges are removed from this runtime model. Each side instead has a narrow
capsule forearm and one actual MuJoCo `ball` joint (3 rotational degrees of
freedom, stored as a 4-component unit quaternion), followed by the Shadow palm
and 18 finger actuators. This is a custom simulated robot, not a hardware-valid
Panda model or an anatomically accurate human wrist.

Five remaining Panda joints position the center of each ball. Three torque
actuators rotate the hand using quaternion/rotation-matrix error feedback;
torque is recomputed every physics step, bounded to 4 Nm in norm. The body has
idealized gravity compensation. The ball has a 130-degree joint limit, while
user orientation targets remain limited to 120 degrees from home and smoothed
at 0.8 rad/s. There are 52 actuators overall. Rotation can occur about the wrist
center without requiring the old terminal Panda links to swing around.

Both keyboard and Vision Pro launchers use this simplified model by default.
Controls, explicit C calibration, stale-data hold, and no-recording behavior
are the same as described below. The default camera shows palms down from
behind the hands. These limits and the small workspace do not implement
collision-aware planning; actual tracking can be constrained by contacts.

Offline verification:

```bash
.venv/bin/python scripts/test_spherical_wrist.py
.venv/bin/python scripts/view_dual_arms.py --headless --demo --seconds 5
```

### Previous Panda wrist assembly (optional)

It attaches each Shadow Hand wrist/finger mechanism to a seven-joint Franka
Panda arm. The redundant 21 cm Shadow forearm housing is replaced at runtime by
a short adapter. Both robot models are fetched by `mujoco-menagerie` on first
use into the ignored `.cache/` directory; both models use Apache-2.0 licenses.
The arms use idealized gravity compensation and position servos, not hardware
torque tuning. The green spheres show commanded wrist positions. The default
operator camera looks forward/down from behind the hands above a worktable;
palms face down and thumbs face inward. Left appears on screen left. This is a
fixed Mac camera view, not a headset-tracked or headset-streamed camera.

Robot commands now live in a separate **Holos — Hand controls** window. Click
that control window before typing these keys, or use its labeled buttons.
MuJoCo handles many of the same letters (I/J/U/P etc.) for visualization before
calling Python, so robot hotkeys are deliberately not registered in its viewer.
Use the simulation window for mouse/camera interaction. This also keeps C
calibration from toggling contact visualization and F from toggling force plots.
The control window uses this Mac's installed Tkinter in a separate normal
Python process, so Tk and mjpython each own their native GUI event loop.
Closing either window stops the demo and closes the other.

- **1 / 2**: select left / right arm (right is initially selected).
- **W / S**: move the selected wrist forward / backward (world X).
- **A / D**: move left / right (world Y).
- **R / V**: move up / down (world Z). Each press changes the target by 1.5 cm.
- **F / O**: close / open both hands in offline mode.
- **I / K**: bend the selected wrist up / down (5 degrees per press).
- **J / L**: tilt the selected hand left / right (5 degrees per press).
- **U / P**: turn the selected palm about the forearm (5 degrees per press).
- **H**: reset both robots to their home poses.
- **B**: switch between the operator view and the wider overview.
- **SPACE**: pause / resume; close the window to stop.

Alternatively run `./run_dual_arms.command --demo` for small scripted arm
movements. This is an offline motion demo, not recorded human demonstrations.

### Live wrist position, orientation, and fingers

Run `run_live_dual_arms.command` and enter the current Tracking Streamer IP or
room code. Keep the headset app active. Hold both hands comfortably, face
forward relative to the headset's tracking-origin frame, and press **C** in the
**Hand controls** window (or click Calibrate) to calibrate position/orientation. After calibration, wrist
**translation and rotation** drive the robot. In the default spherical mode,
position IK moves the wrist center and torque control rotates the ball joint.
In optional `--wrist panda` mode, two Shadow hinges handle bending/deviation,
with the remaining rotation handled by the seven Panda joints; that mode's
pose reference is the palm base. The finger-bend mapping controls only fingers
in both modes, so opening/closing fingers does not reset wrist commands.
The coordinate convention is SDK +Y forward -> robot +X, SDK +X right -> robot
-Y, SDK +Z up -> robot +Z. C resets position offsets, not the yaw convention.
Rotation uses the relative 3D orientation since C and the same SDK-to-world
axis conversion, rather than raw Euler angles. Recalibrating aligns the current
human orientation with the current simulated palm orientation. If translation
directions are rotated, face the tracking-origin forward direction before
testing. This coordinate convention still needs validation with a live headset.

The default translation scale is 0.6 (10 cm human motion -> 6 cm robot target
motion). Orientation follows at 1:1 scale, capped at 120 degrees from the robot's
home orientation. Orientation targets change at no more than 0.8 rad/s, and
in optional Panda mode, the two wrist actuator commands change at no more than
0.6 rad/s within their original limits. Panda IK handles residual rotation when
a hinge limit is met. Spherical mode instead uses the torque limits above.
These are prototype limits, not calibrated human wrist measurements.
The position target is clipped to a small volume around each robot's home
wrist: +/-16 cm in X, +/-14 cm in Y, -16/+18 cm in Z. Arm commands are limited
to 0.6 rad/s and joint limits; these bounds are not collision-aware planning.
Actual position/orientation can lag or stop short at unreachable targets. The
green marker and its small red/green/blue axes show the commanded reference
point and smoothed orientation; they do not certify successful tracking.

Press **C** again to recenter. Stale tracking (>0.5 seconds), pause, or H reset
invalidates arm calibration; fresh tracking resumes finger control but the arm
requires another C. If both hands are stale, the entire simulation freezes;
if one is stale, that side keeps its last joint command while the other runs.
The SDK's previously noted repeated-frozen-skeleton limitation still applies.

This version shows the simulation on the Mac and does not record episodes
(recording is only available in the human-arm workspace; see "Recording
episodes"), train a policy, or stream the rendered scene into the headset. An offline test
does not verify a live headset connection or real-world grasping performance.

Offline checks (no headset or graphical viewer required):

```bash
.venv/bin/python scripts/test_arm_control.py
.venv/bin/python scripts/test_control_panel.py
.venv/bin/python scripts/view_dual_arms.py --headless --demo --seconds 5
```

Optional preview PNG (requires a Mac graphical session/OpenGL):

```bash
.venv/bin/mjpython scripts/view_dual_arms.py --headless --demo --seconds 5 \
  --screenshot outputs/dual_arms_preview.png
```

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
# Human-shaped arm variant

Both human-arm launchers now include a tabletop workspace: fixed table and
teal tray, plus a yellow hollow cup with a handle (0.21 kg, free to move).
Use H in the separate controls panel to reset the arms AND cup. In live
mode recalibrate with C afterwards. The cup has gravity; only the arms
use ideal gravity compensation. Grasping and placement are not yet
validated with live headset input; F is a generic fist, not an automatic
grasp command. No liquid, breakage or real-material fidelity is modeled.
Scene dimensions and initial positions live in `scripts/workspace_scene.py`.

**Scene layout (version 2).** The worktop top is at z = 0.60 m (was 0.505),
the cup starts at (0.30, −0.27) and the tray is centred at (0.30, −0.09)
(was (0.32, −0.28) and (0.43, −0.12)); x is forward, y is the robot's left.
At the old height the right palm could only rest on the cup's rim, because
the wrist clip box and 0.53 m reach cap stop it about 6 cm too high. The arms,
clip box and reach cap are unchanged; only the props moved. Episodes store
`metadata.scene.scene_version` (missing = version 1) and the prop positions,
so older episodes still label correctly; replay warns that their props are
drawn at the new positions.

`scripts/check_reach.py` is a headless probe of that workspace. It sweeps
each wrist over its clip box (2 cm grid, palm-down / palm-inward / palm-inward
pitched 30°), settles physics, and reports the best palm-inward wrap of the
cup and whether a cup held that way can be put in the tray with 3 cm to
spare from every limit (about 3 minutes on 11 cores; `--side right`,
`--json out.json`). Re-run it after moving props. With the Shadow hand turned
palm-inward the open thumb rests on the cup's rim, so the report also counts
those "rim" wraps. Version 2 result: right-hand wrap with 5.6 cm limit
clearance, placement at the tray centre with 5.7 cm.

`scripts/scripted_cup_to_tray.py` is a deterministic right-hand pick and
place built only from controller targets and the gradual open/close. It
records a `scripted_demo` episode that labels SUCCESS (reach > grasp > lift >
transport > place > release > retreat). It shows the scene is solvable within
the live limits; it is not a policy. Add `--viewer` (with `mjpython`) to
watch it.

`run_human_arms.command` opens an offline shoulder–elbow–wrist prototype.
`run_live_human_arms.command` connects the same model to Tracking Streamer:
press Start on Vision Pro, enter its current IP or room code on the Mac,
then use C in the separate controls window to calibrate a comfortable pose.

Each fixed shoulder is a three-axis ball joint, the elbow is a one-axis
hinge, and the wrist is a three-axis ball joint connected to Shadow Hand.
This variant uses ideal gravity compensation and is not a hardware-faithful
robot or anatomical model. The human variant now accepts full 180-degree
orientation targets, with no ball-joint cone limit on its virtual wrist;
this does not represent anatomical wrist bending or a physical robot's range.
Older Panda/spherical variants retain their original limits.

### Optional forearm input

The installed `avp_stream` SDK exposes wrist-local `left_arm`/`right_arm`
arrays: 25 points on older streams, or 27 with `forearmWrist` (25) and
`forearmArm` (26). The bridge validates those two extra transforms and their
separation (8–50 cm), and rotates their difference into the SDK world frame.
It does NOT treat these local arrays as already-world-space HandData.

Press C with both hands visible and held comfortably. This calibrates
relative wrist motion AND a direction-only forearm alignment. The controller
uses two-bone geometric IK: wrist reach determines an elbow circle, and
forearm direction guides a bounded swivel on that circle. Both elbows
start slightly outward. Away from the center, a 15 mm outward shoulder-plane
margin overrides inward forearm requests. For hands meeting at the center,
the margin relaxes: elbows may come inside shoulder width, but remain at
least 8 cm on their own side of the body centerline. Infeasible extreme
reaches are shortened toward home. This posture envelope is not general
collision avoidance. Fixed robot
shoulders and limb lengths mean this is not a one-to-one copy of your elbow
position; conflicting or unreachable directions are approximated.

The controls panel reports per side:

- `forearm ON`: usable current vector and calibrated direction guide.
- `forearm press C`: usable data, but no forearm calibration yet.
- `forearm unavailable`: missing/invalid/stale data; wrist-only fallback.

Missing optional forearm data never discards valid finger or wrist data.
The SDK's arrays do not expose per-joint confidence flags, so the validity
checks cannot prove measurement accuracy or distinguish every repeated
placeholder from a genuinely stationary limb. Wrist tracking loss, pause,
and home reset still require C to recalibrate. Forward/vertical translation
remains at 0.6 scale. Human-arm lateral workspaces now overlap (30 cm inward,
14 cm outward from home). When both hands are fresh during C, lateral gain
is fitted to your initial left/right wrist spacing, so bringing real wrists
together does not retain the old artificial 50 cm anchor gap. Face forward,
hold wrists comfortably 20–70 cm apart, and calibrate both together. Implausible
spacing/gain falls back to the ordinary scale; this is not body-heading
calibration. The wrist housing radius is 18 mm and the forearm sleeve stops
3 cm short of the pivot so a bulky casing does not prevent palm contact.
All collision geometry remains enabled.

Offline tests exercise packet decoding, 25-point fallback, calibration,
elbow swivel with the wrist held in place, and physical 180-degree palm
rolls. Run `.venv/bin/python scripts/test_forearm_tracking.py`.
Human-arm command limits are now 1.8 rad/s for shoulder/elbow and 2.5 rad/s
for palm orientation and forearm-guide smoothing. The viewer budgets physics
steps from elapsed wall time rather than a fixed 10 ms per rendered frame.
Individual physics steps stay at 2 ms; long stalls are capped at 50 ms of
catch-up to avoid a large jump. Paused/stale time is not replayed on resume.
`scripts/test_posture_timing.py` checks mirrored home, inward-guide rejection,
30/60/100/144 Hz timing, and physical response at a 30 Hz render budget.
Live forearm availability/quality and the resulting visual correspondence
still need headset validation; no live connection is claimed by these tests.
`scripts/test_bimanual_contact.py` verifies lateral span fitting, invalid-span
fallback, physical palm-to-palm contact (not just wrist contact), and release.

### Gradual offline opening and reach feedback

In the arm viewer, F now eases both hands toward the fist command over 0.9 s,
and O eases toward open over 1.3 s, using a cubic ramp at each 2 ms physics
step. Repeated keys do not restart an identical transition. Pause freezes
the ramp and H clears it together with the scene reset. Live finger mapping
is unchanged. This reduces abrupt command changes; it does not guarantee
that every cup/handle contact configuration releases without an impulse.
The immediate `set_fist` helper remains available to offline tests only.

Human-arm status shows `REACH LIMIT` if the requested wrist position is
clipped by more than 5 mm by workspace, arm-length or elbow-envelope limits.
This is not a collision warning. Calibration C recenters the input; it
cannot move a fixed shoulder or extend the 28 cm + 26 cm arm. The table, cup
and tray were moved into reach (scene version 2, above). H resets both arms
and the cup if the cup has been thrown outside reach. No movable base or torso is implemented here.
`scripts/test_hand_transition.py` checks ramp timing, interruptions, repeated
keys, reset and reach warnings; a live/user grasp-and-release test is still
needed to confirm the cup no longer gets pushed during opening.

## Recording episodes

The human-arm workspace scene (`--wrist human --workspace`, i.e. both
`run_human_arms.command` and `run_live_human_arms.command`) can record
episodes. An episode is **simulated robot state driven by human tracking**
(or keyboard/scripted targets), not real-robot data. Live episodes also keep
the raw Vision Pro input, so they can be re-run through a future retargeter.
Keyboard and live episodes share one schema but are not equivalent
demonstrations.

| Key / button (controls window) | Action |
|---|---|
| **E** · Record | Start an episode; press again to stop and save it |
| **X** · Discard | Stop and delete the current episode |
| **H** | Save the current episode (`end_reason="reset"`), then reset arms and cup |
| Space | Pause/resume; no samples are recorded while paused |
| Esc / close a window | Save the current episode (`end_reason="viewer_closed"`) |

The status line shows `● REC 00:12.4 · ep 0007` while recording, then
`Saved ep 0007 · SUCCESS` or `Saved ep 0007 · FAIL (not_released)`; the
terminal prints the saved folder. Episodes auto-stop at
`--max-episode-seconds`. In live mode press C before E; starting earlier is
allowed but warns `REC started before calibration (press C)`.

Flags (the launchers forward extra arguments):

| Flag | Default | Meaning |
|---|---|---|
| `--record-dir` | `data/episodes` | Where episodes are written (git-ignored) |
| `--task` | `cup_to_tray` | Task id; the labeler evaluates `cup_to_tray` only |
| `--operator` | empty | Optional tag such as initials (≤ 32 characters) |
| `--video` / `--no-video` | on | Render operator + overview camera MP4s |
| `--video-fps` | 30 | Frame rate in simulation time |
| `--video-size` | `640x480` | Per-camera size (even numbers) |
| `--max-episode-seconds` | 120 | Auto-stop and save |
| `--no-auto-label` | off | Skip labeling after saving |

For example: `./run_live_human_arms.command --operator NT --video-fps 15`.

**Tips for live recording**

- Press **H** then **C** before each episode, so the arms and cup start at home
  and the mapping is fresh.
- Keep your hands within about 25 cm of where they were when you pressed C.
  At the default `--scale 0.6` that already covers the arm's clip box.
- Watch the status for `REACH LIMIT`. It means the requested wrist position
  is clipped, so move back toward the calibration pose.
- If a side shows `arm press C`, its tracking dropped for more than 0.5 s and
  that arm is frozen until you press **C**.
- On flaky Wi-Fi add `--auto-recenter`, for example
  `./run_live_human_arms.command --auto-recenter`. An arm that was following
  and whose tracking returns within `--recenter-window` seconds (default 3)
  re-anchors at its current pose with its previous gains, shows
  `arm FOLLOW (auto)` for a second, and logs an `auto_recenter` event.
  Longer gaps still need C. The default (freeze until C) is unchanged.
- To see the simulation in the headset instead of on the Mac, add
  `--headset-view`, for example
  `./run_live_human_arms.command --auto-recenter --headset-view`.

- `--grasp-assist` (off by default; live and keyboard modes) helps a
  marginal grasp hold: see "Grasp assist" below.

**Grasp assist.** When the thumb and at least one other finger touch the cup
and your commanded grip is closed past about 50 % and has stopped closing,
a weld constraint holds the cup to that palm at its current pose (the cup is
never moved). It lets go when the grip opens below about 35 %, when the hand
has not touched the cup for 0.3 s, or on H/reset. The terminal prints
`Grasp assist ON/OFF`. Episodes recorded with it are flagged, not hidden:
`assist/active` per sample, `grasp_assist` in metadata, `assist_on`/
`assist_off` events and an `assist` column in `summarize_episodes.py`. The
labeler ignores the flag, so assisted grasps are labelled like any other;
filter on the flag when picking demonstrations for training. The welds exist
(inactive) in every workspace model, so older episodes still replay.

Grip physics: every cup contact uses condim 4 (torsional friction 0.005), so a
pinched cup does not spin about the contact; sliding friction, size and mass
are unchanged, and the cup still slides when pushed and tips when knocked.
`.venv/bin/python scripts/live_grasp_check.py [--grasp-assist]` closes the
right hand at the scripted wrap pose only through the live finger mapping,
lifts 5 cm, holds 2 s and prints lift, slip and tilt (no headset needed).

**Headset view.** `--headset-view` streams the MuJoCo scene to the Vision
Pro as a floating **2D video window** through Tracking Streamer's WebRTC
video (the same IP or room code). It is not AR: nothing is placed in your
room, and latency depends on your Wi-Fi. The window appears once the headset
connects. Hand tracking and recording keep working if it never does, or if
the stream fails (the terminal prints one line).

- **What it shows:** by default (`--headset-camera follow`) whatever the Mac
  viewer camera shows. To adjust the view, move the Mac camera with the mouse
  or press B for operator/overview. `operator` and `overview` lock the stream
  to the fixed workspace views instead.
- **Overlay:** a small strip at the bottom left shows each hand's
  `LIVE`/`WAIT` and `FOLLOW`/`press C` (plus `REACH` and `(auto)`), a red
  `REC mm:ss` while recording, and the last saved episode's outcome, so you
  can tell when to calibrate (C) and whether recording (E) is on without
  looking at the Mac. The C/E buttons are still in the Mac's Hand controls
  window: someone at the Mac presses them.
- **Cost:** defaults are `--headset-size 960x540 --headset-fps 20`.
  Each streamed frame is an extra offscreen render on the control thread.
  A frame that costs more than `--headset-budget-ms` (default 16) skips the
  next slot. Measured on the development Mac with nothing else running, while
  recording 30 fps video (loop period median / p95):

  | Headset view | Loop (ms) | Stream | Render | Recorder frames dropped |
  |---|---|---|---|---|
  | off | 12.0 / 13.8 | – | – | 2 |
  | `follow` 960x540 @ 20 (default) | 13.2 / 28.4 | 17.7 fps | 13.8 ms | 12 |
  | `operator` 960x540 @ 20 | 12.9 / 15.2 | 17.6 fps | 2.0 ms (reused) | 2 |
  | `follow` 1280x720 @ 30 | 13.6 / 30.6 | 18.4 fps | 14.8 ms | 60 |

  With `--headset-camera operator` and recording on, the recorder's operator
  frame is reused (letterboxed, 640x480) instead of rendering twice, so it
  is the cheapest choice. 1280x720 @ 30 goes over budget on almost half its
  frames and drops five times as many recorder frames, so it is not the
  default. With another simulation running, renders took about 17.6 ms and
  the default stream fell to about 9 fps.
  `.venv/bin/mjpython scripts/bench_headset_view.py` measures this on your Mac;
  close other simulations first.

Each episode is a folder `data/episodes/<YYYYMMDD-HHMMSS>_<task>_<NNNN>/` with
`episode.h5`, `episode.json` (the same metadata as the HDF5 `metadata`
attribute) and, with video, `cam_operator.mp4` and `cam_overview.mp4`.
`episode.h5` (`schema="lab3l.episode"`, `schema_version=2`) is sample-major,
with one sample per loop iteration in which physics advanced (~100 Hz):

| Group | Contents |
|---|---|
| `time/` | `sim` (from 0, strictly increasing) and `wall` seconds |
| `robot/` | `qpos`, `qvel`, `ctrl`, actual and target wrist pose, `reach_error` |
| `input/` | raw wrist transform, 25-joint finger skeleton, 27-point arm skeleton, forearm vector, retargeted finger targets, `fresh`, packet age/sequence. NaN/False/-1 in keyboard mode |
| `scene/` | cup pose (xyz + wxyz) and linear/angular velocity |
| `contacts/` | hand–cup per finger group (thumb, ff, mf, rf, lf, palm, other), arm–cup, hand–cup normal force, cup–table, cup–tray, hand–hand, `ncon` |
| `assist/` | `active` (N, 2) left/right: the `--grasp-assist` weld holds the cup. All False without the flag; absent in older episodes |
| `video/<camera>/` | `frame_time` and `sample_index` of every written frame |
| `labels/` | per-sample `phase`, plus segments/events/outcome attributes |

Root attribute `events` is a JSON list of record start/stop, pause/resume,
reset, calibrate (with the wrist-mapper calibration), auto_recenter (side,
gap), tracking lost/regained, reach-limit on/off, assist_on/assist_off (side,
and closure or reason) and errors. Metadata records `grasp_assist` (flag on or
off) and `grasp_assist_activations`, and the model's joint/actuator
names, camera intrinsics, software versions and git commit. It never stores
the Tracking Streamer address or absolute paths. Do not commit recordings.

**Video caveats.** Frames are rendered offscreen on the control thread, with
one camera per loop iteration, and encoded on a background thread. If the
encoder falls behind, frames are dropped and counted in
`metadata.video.dropped_frames`; `frame_time` keeps exact alignment. Measured on
an M-series Mac with the viewer open (scripted demo, 640x480, 30 fps): about
11 ms per render whatever the resolution, and 12 ms median / 14 ms max loop
work, against 2 ms without video. If control feels sluggish, use
`--video-fps 15` or `--no-video`. If the renderer cannot be created, recording
continues state-only and `metadata.video.status` says why.

**Labels.** After each save, `scripts/episode_labels.py` runs automatically. It
splits the episode into phases (idle, reach, grasp, lift, transport, place,
release, retreat, recover) and decides cup-to-tray success with a failure
reason (`never_touched`, `never_lifted`, `dropped_outside_tray`, `tipped_over`,
`not_in_tray`, `not_released`, `not_settled`, `too_short`). It uses only the
recorded contacts and cup motion. Thresholds are hand-chosen heuristics in
`LabelConfig`, not yet tuned against human-labelled episodes. Confidence
scores are heuristic too. The three confidence factors are stored per
segment so the labeler can be recalibrated later.

```bash
.venv/bin/python scripts/label_episode.py data/episodes            # (re)label all; --dry-run to preview
.venv/bin/python scripts/summarize_episodes.py --csv episodes.csv  # outcome/phase/assist table
.venv/bin/mjpython scripts/replay_demo.py data/episodes/<id>/episode.h5
```

Replay rebuilds the workspace scene, checks that joint/actuator names match,
restores arms, hands and cup, and prints phase transitions and the outcome.
Tests: `scripts/test_episode_recorder.py`, `scripts/test_episode_labels.py`,
`scripts/test_scripted_cup_to_tray.py`, `scripts/test_auto_recenter.py`,
`scripts/test_check_reach.py`, `scripts/test_headset_view.py`, `scripts/test_live_grasp.py`.


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
physics from actions. Files must match the default single left Shadow Hand model,
except recorded `lab3l.episode` files (see "Recording episodes").

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
