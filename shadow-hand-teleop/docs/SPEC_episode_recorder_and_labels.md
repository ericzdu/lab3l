# Spec: Live episode recorder + automatic success/phase labels

**Owner:** Nick Tiesi · **Branch:** `Nick` (from `origin/Emma`) · **Status:** ready to implement
**Scope:** `shadow-hand-teleop/` only. Target the human-arm workspace scene
(`view_dual_arms.py --wrist human --workspace`): two arms, Shadow Hands, cup, tray.

## 0. Why

TEAM_SETUP.md §6 lists "No live arm-demo recorder" as an open gap: every live session
ends with "No data saved". The Vision Pro is available to one teammate per week, so
each headset session must produce reusable data. This change adds:

1. **Recorder (Part A):** start/stop episodes from the controls window. Each episode
   saves raw Vision Pro input, simulation state, cup/contact signals, events and
   rendered camera video, in a versioned HDF5 file the existing replayer can play.
2. **Labeler (Part B):** an offline, pure-function pass over a recorded episode that
   decides cup-to-tray success (with a failure reason) and segments it into
   manipulation phases with confidence scores. It runs automatically after each save
   and as a CLI, so teammates can improve it later without a headset.

Recording **raw** tracking (not only retargeted commands) is a hard requirement:
finger retargeting will change (for example dex-retargeting), and old demos must be
re-runnable through new retargeters.

## 1. Ground rules for the implementer

- Before editing, run `git status` / `git branch`. If not already on `Nick`, run
  `git fetch origin && git switch -c Nick origin/Emma` (or `git switch Nick` if it
  exists). Never commit to `Emma` or `main`. Do not push; Nick will push.
- Read `TEAM_SETUP.md`, `README.md`, `scripts/view_dual_arms.py`,
  `scripts/teleop_shadow_hands.py`, `scripts/control_panel.py`,
  `scripts/replay_data.py`, `scripts/replay_demo.py`, `scripts/workspace_scene.py`,
  `scripts/human_arms.py`, `scripts/arm_control.py` first. Match the existing style:
  small modules in `scripts/`, `unittest` tests named `test_*.py`, metres/radians/kg,
  honest docstrings about limitations.
- **Keep the control loop's behaviour and edits small.** Put new logic in new
  modules; `view_dual_arms.py` gets only hook calls. Existing launchers, mock export,
  NPZ/flat-HDF5 replay and HolosCapture replay must keep working unchanged.
- **Never block or slow the control loop.** Video encoding happens off-thread; if the
  encoder falls behind, drop frames and count them.
- **Privacy (TEAM_SETUP §5/§8):** never write the Tracking Streamer IP/room code,
  usernames or absolute home paths into recordings, logs or metadata. Recordings go
  under `data/` (already git-ignored) and are never committed.
- New dependencies: only `imageio` and `imageio-ffmpeg` (MP4 encoding), plus `Pillow`
  if it isn't already pinned (it is used by `save_screenshot`). Add them to
  `requirements.txt` with version bounds.
- Run the full suite before finishing:
  `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`.

## 2. User-facing behaviour

### Controls (both live and keyboard mode, workspace scene only)

| Key / button | Action |
|---|---|
| **E** · "Record · E" | Start a new episode; press again to stop and save it. |
| **X** · "Discard · X" | While recording: stop and delete the current episode (nothing kept). |
| **H** (existing) | If recording: stop and **save** the episode (`end_reason="reset"`), then reset as before. |
| Space (existing) | Pause/resume. While paused no samples are appended; pause/resume are logged as events. |
| Esc / closing either window | If recording: save the episode (`end_reason="viewer_closed"`). |

- Add both buttons to `control_panel.py`, add `E` and `X` to `KEYS`, and add them to
  the live-mode allow-list (currently `"12HCB "`). Both keys are free today (robot
  keys use W/S/A/D/R/V/I/K/J/L/U/P/F/O). Disable the buttons when not in the
  workspace scene.
- Status line additions: `● REC 00:12.4 · ep 0007` while recording; after saving,
  `Saved ep 0007 · SUCCESS` or `Saved ep 0007 · FAIL (not_released)`. Also print the
  saved path (relative to the project) to the terminal.
- Auto-stop at `--max-episode-seconds` (default 120) with `end_reason="max_duration"`.
- Recording needs the cup scene: if `E` is pressed without `--workspace --wrist human`,
  show `Recording requires --wrist human --workspace` and do nothing.
- Pressing `E` in live mode before both arms show `arm FOLLOW` is allowed, but show a
  one-time warning in the status line: `REC started before calibration (press C)`.

### New CLI flags on `view_dual_arms.py`

| Flag | Default | Meaning |
|---|---|---|
| `--record-dir` | `data/episodes` (project-relative) | Where episodes are written |
| `--task` | `cup_to_tray` | Task id stored in metadata (labeler uses it) |
| `--operator` | `""` | Free-text operator tag, e.g. initials. No validation beyond length ≤ 32 |
| `--video / --no-video` | on | Render camera frames |
| `--video-fps` | 30 | Rendered frame rate (sim time) |
| `--video-size` | `640x480` | Per-camera resolution |
| `--max-episode-seconds` | 120 | Auto-stop |
| `--no-auto-label` | off | Skip Part B after save |

The existing `.command` launchers already forward `"$@"`, so no launcher changes are needed.
Do not add a new launcher.

## 3. Part A — Recorder

### 3.1 New module `scripts/episode_recorder.py`

```python
class EpisodeRecorder:
    def __init__(self, model, controller, hands, *, record_dir, task, operator,
                 source,               # "live" | "keyboard" | "scripted_demo"
                 wrist_mode, workspace, scale, video=True, video_fps=30,
                 video_size=(640, 480), max_seconds=120.0): ...
    recording: bool
    def start(self, data) -> Path              # creates episode dir, opens video writers
    def sample(self, data, *, samples, fresh, paused, mapper) -> None
        # call once per loop iteration AFTER physics advanced (steps > 0)
    def maybe_render(self, data) -> None       # main thread; renders when sim time crosses next frame time
    def event(self, data, kind, **payload) -> None
    def stop(self, data, end_reason) -> Path   # flush, write HDF5 + JSON, close video, return episode dir
    def discard(self) -> None
```

- Buffers state samples in memory (lists of `float32`/`float64` arrays). An episode of
  120 s at ~100 Hz with raw skeletons is under ~100 MB; that is acceptable.
- `sample()` is only called on iterations where `steps > 0`, so paused or frozen
  frames produce no duplicate samples. Timestamps come from `data.time` (sim time) plus
  `time.monotonic()` for wall time. Both are rebased so the first sample is t=0.
- The episode directory is `data/episodes/<YYYYMMDD-HHMMSS>_<task>_<NNNN>/` where
  `NNNN` is a zero-padded counter (the next free one in `record_dir`).

### 3.2 Changes to `TrackingBridge` (`teleop_shadow_hands.py`)

The recorder needs raw input. When `with_wrist=True`, **append** fields to the
per-side tuple (do not reorder existing indices 0–3, which other code relies on):

| idx | existing / new | content |
|---|---|---|
| 0 | existing | `time.monotonic()` at receipt |
| 1 | existing | finger target dict |
| 2 | existing | wrist 4×4 (validated) |
| 3 | existing | forearm vector or `None` |
| 4 | **new** | raw finger skeleton `sample[f"{side}_fingers"]` as `(25,4,4)` float array copy |
| 5 | **new** | raw 27-point arm skeleton if present, else `None` (copy) |
| 6 | **new** | monotonically increasing packet sequence number (bridge-wide int) |

Check `test_forearm_tracking.py` and other tests for tuple-length assumptions and update
them if needed. The `with_wrist=False` path (the original finger-only demo) is unchanged.

### 3.3 Integration in `view_dual_arms.py`

Minimal hooks only:
- Construct `EpisodeRecorder` after `setup()` when `--workspace` is set. Set
  `model.vis.global_.offwidth/offheight` ≥ the video size **before** creating any
  renderer.
- Event queue handling: `E` → start/stop, `X` → discard, `H` → stop(save) then the
  existing reset, Space/C → `recorder.event(...)`, quit → stop(save) in `finally`.
- After the physics block (`if steps: ...`): if recording, call `recorder.sample(...)`
  then `recorder.maybe_render(data)`.
- On `C`, log a `calibrate` event whose payload is a JSON-safe snapshot of the
  `WristMapper` state (`origins`, `anchors`, `rotation_origins`, `rotation_anchors`,
  `position_scales`, `forearm_alignments` for each calibrated side; arrays → lists) and
  whether the hand-span calibration succeeded.
- Also log events for: `record_start`, `record_stop`, `pause`, `resume`, `reset`,
  `tracking_lost`/`tracking_regained` per side (edges of `fresh`), `reach_limit_on/off`
  per side (edges of `reach_errors > .005`), `error` (with message).
- Update the print lines that say "No data saved"/"No data recording is enabled" so
  they are accurate (for example `Press E to record an episode.`).

### 3.4 Per-sample data (HDF5 layout, `episode.h5`)

Root attrs: `schema = "lab3l.episode"`, `schema_version = 2`, `metadata` (JSON string,
§3.6). Datasets are gzip-compressed, sample-major, N = number of samples. Sides are
always ordered `("left","right")`.

```
time/sim                      (N,)        float64  s, from 0, strictly increasing
time/wall                     (N,)        float64  s since start (monotonic)

robot/qpos                    (N,nq)      float64
robot/qvel                    (N,nv)      float64
robot/ctrl                    (N,nu)      float64  (data.ctrl after the step = commanded)
robot/wrist_pos               (N,2,3)     float64  actual wrist_site position
robot/wrist_quat              (N,2,4)     float64  actual wrist_site orientation (wxyz)
robot/target_pos              (N,2,3)     float64  controller.targets
robot/target_quat             (N,2,4)     float64  controller.rotation_targets → wxyz
robot/reach_error             (N,2)       float32  controller.reach_errors (0 if absent)

input/fresh                   (N,2)       bool     same freshness rule as the loop (<0.5 s)
input/packet_age              (N,2)       float32  s since packet receipt (NaN if none)
input/packet_seq              (N,2)       int64    -1 if none
input/new_packet              (N,2)       bool     seq changed since previous sample
input/wrist                   (N,2,4,4)   float32  raw validated wrist transform (NaN if none)
input/finger_skeleton         (N,2,25,4,4) float32 raw wrist-local skeleton (NaN if none)
input/arm_skeleton            (N,2,27,4,4) float32 raw arm skeleton (NaN if absent)
input/forearm_vector          (N,2,3)     float32  NaN if None
input/finger_targets          (N,2,K)     float32  retargeted targets in FINGER_TARGET_NAMES order
                                                     (attr `names`), NaN if none

scene/cup_pose                (N,7)       float64  xyz + wxyz quat (body "cup")
scene/cup_linvel              (N,3)       float32  world linear velocity (mj_objectVelocity)
scene/cup_angvel              (N,3)       float32
contacts/hand_cup             (N,2,7)     bool     per side × group: thumb, ff, mf, rf, lf, palm, other_hand
contacts/arm_cup              (N,2)       bool     any arm link (shoulder/elbow/wrist bodies) touching cup
contacts/hand_cup_force       (N,2)       float32  summed normal force, N (mj_contactForce)
contacts/cup_table            (N,)        bool     cup vs worktop/legs
contacts/cup_tray             (N,)        bool     cup vs any tray geom
contacts/hand_hand            (N,)        bool     left hand/arm vs right hand/arm
contacts/ncon                 (N,)        int32

video/<camera>/frame_time     (F,)        float64  sim time of each written frame
video/<camera>/sample_index   (F,)        int64    index into time/sim at render time

events                        JSON string attr: [{"t": sim_time, "wall": ..., "kind": ..., ...}]
```

In keyboard/scripted mode, all `input/*` arrays exist but are NaN/False/-1, so the
schema is identical across sources.

**Contact grouping:** precompute geom→group lookup tables once at init from body names.
Hand bodies are named `"{side}_{s}h_<part>"` (for example `left_lh_ffdistal`). Map
`<part>` prefixes `th*`→thumb, `ff*`→ff, `mf*`→mf, `rf*`→rf, `lf*`→lf, `palm`→palm,
anything else on the hand→other_hand. Arm bodies are `"{side}_shoulder"`,
`"{side}_elbow"`, `"{side}_wrist"`. Cup geoms belong to body `cup`; tray geoms to body
`tray`; table geoms are `worktop` and `table_leg_*`. Assert at init that each lookup
is non-empty and fail loudly with the missing names if the scene changes.

### 3.5 Video

- Two **free cameras** (`MjvCamera`, no model change, so nq/nv/nu stay identical):
  `operator` = `configure_camera(cam)` and `overview` = `configure_camera(cam,
  overview=True)`. Reuse that function so the views match the viewer.
- One `mujoco.Renderer` created at the first `start()` and reused across episodes
  (do not create one per frame or per episode). Render on the main thread inside
  `maybe_render()` whenever `data.time ≥ next_frame_time`. Hand `(H,W,3) uint8` copies
  to a background writer thread through a bounded queue (size ~60). If the queue is
  full, drop the frame and increment `dropped_frames[camera]`.
- The writer thread encodes `cam_<name>.mp4` with `imageio` (libx264, `quality=7` or
  `-crf 23`, `-preset veryfast`, `pix_fmt=yuv420p`, fps = `--video-fps`). Record each
  written frame's sim time and sample index so labels can be aligned to video frames
  exactly, even with drops.
- Store camera parameters in metadata: lookat, distance, azimuth, elevation,
  `model.vis.global_.fovy`, width, height, and the derived pinhole intrinsics
  (fx = fy = (H/2)/tan(fovy/2), cx = W/2, cy = H/2).
- **Risk to verify on macOS/mjpython:** creating a `mujoco.Renderer` while
  `launch_passive` is open. If it fails, catch the exception, keep recording
  state-only, set `metadata.video.status = "unavailable: <error>"`, and show it once in
  the status line. Never crash the session over video. If rendering at 30 fps visibly
  degrades control (loop period > 15 ms), document the measured numbers in the PR.
  Then lower the default to 15 fps or 480×360 and note it in the README.

### 3.6 Metadata (`metadata` attr **and** `episode.json` sidecar, same content)

```json
{
  "schema": "lab3l.episode", "schema_version": 2,
  "episode_id": "20261027-153012_cup_to_tray_0007",
  "task": "cup_to_tray", "source": "live", "operator": "NT",
  "created_utc": "...", "duration_sim_s": 18.42, "samples": 1840,
  "end_reason": "user_stop | reset | viewer_closed | max_duration | error",
  "scene": {"wrist_mode": "human", "workspace": true, "scale": 0.6,
            "cup_start": [...], "tray_center": [...], "table_top": 0.505},
  "model": {"nq": .., "nv": .., "nu": .., "timestep": ..,
            "joint_names": [...], "actuator_names": [...]},
  "software": {"mujoco": "3.15.0", "avp_stream": "2.51", "python": "3.12.x",
               "git_commit": "<short sha or null>", "git_dirty": true},
  "finger_target_names": [...],
  "video": {"status": "ok", "fps": 30, "cameras": {"operator": {...}, "overview": {...}},
            "dropped_frames": {"operator": 0, "overview": 0}},
  "labels": null
}
```
`labels` is filled by Part B (§4.4). Get `git_commit` with
`git rev-parse --short HEAD` (handle a missing git gracefully). Never include the
tracking address or absolute paths.

### 3.7 Replay support

- `replay_data.load_demo()` detects `schema == "lab3l.episode"` and returns
  `timestamp`=`time/sim`, `qpos`, `qvel`, `action`=`robot/ctrl`, plus the parsed
  metadata (for example under a `"_metadata"` key, or by returning a small dataclass).
  Keep the existing return shape for the other formats.
- `replay_demo.py`: for episode files, build the model with
  `build_human_model(workspace=True)` (from metadata `scene`), check
  `joint_names`/`actuator_names` match exactly (clear error otherwise), then use the
  existing restore/playback loop. If `labels` exist, print phase transitions to the
  terminal as playback crosses them, and the final outcome at the end.
  Optional: show the current phase with the viewer's text overlay if available in
  MuJoCo 3.15.
- `--headless` must work for episodes (used by tests).

## 4. Part B — Success + phase labeler

### 4.1 New module `scripts/episode_labels.py`

A **pure function** over loaded arrays and metadata, with no MuJoCo stepping, so
it is fast, testable with synthetic data, and re-runnable on old episodes:

```python
@dataclass(frozen=True)
class LabelConfig:            # every threshold lives here and is saved with the labels
    lift_height: float = 0.010        # m above the cup's start height
    settle_window: float = 0.5        # s at end of episode
    settle_lin_speed: float = 0.02    # m/s
    settle_ang_speed: float = 0.5     # rad/s
    upright_max_deg: float = 25.0
    debounce: float = 0.10            # s; minimum duration for a contact/support state change
    drop_min_fall: float = 0.02       # m of descent while unsupported & untouched
    reach_speed: float = 0.05         # m/s wrist speed toward cup

def label_episode(arrays: dict, metadata: dict, cfg=LabelConfig()) -> dict: ...
```

Use the scene constants from `workspace_scene.py`. **Refactor** the tray's literal
sizes there into module constants (`TRAY_HALF_EXTENTS = (.10, .075)`,
`TRAY_RIM_THICKNESS = .005`, `CUP_RADIUS = .034`, `CUP_HALF_HEIGHT = .057`) without
changing the generated geometry, and import them. A test must assert the geometry is
unchanged (same geom sizes/positions as before).

### 4.2 Derived per-sample signals (all debounced by `cfg.debounce`)

- `touch[side]` = any `contacts/hand_cup[:, side, :]` or `contacts/arm_cup[:, side]`.
- `opposition[side]` = thumb contact AND (any of ff/mf/rf/lf or palm). This is a
  proxy for a stable grasp.
- `supported` = `cup_table` or `cup_tray`.
- `lifted` = not supported and cup z > start z + `lift_height`.
- `active_side` = side with the most `touch` samples (ties → right). Bimanual
  episodes (both sides `opposition` for > 0.3 s simultaneously) set `bimanual=True`
  and still choose the side with more contact time.

### 4.3 Phases (state machine, one label per sample)

Codes, stored as an int8 array with a `names` attr:
`0 idle, 1 reach, 2 grasp, 3 lift, 4 transport, 5 place, 6 release, 7 retreat, 8 recover`.

- **idle → reach:** active wrist speed toward the cup > `reach_speed` (projected
  velocity onto the wrist→cup direction), sustained ≥ debounce.
- **reach → grasp:** first debounced `touch[active]`.
- **grasp → lift:** cup becomes debounced `lifted`.
- **lift → transport:** cup horizontal speed > 0.05 m/s or horizontal displacement
  from lift-off point > 0.03 m.
- **transport → place:** first debounced `cup_tray` while still `touch[active]`.
  (If the cup touches the tray with no hand contact, that is not a place. See drop.)
- **place → release:** `touch[active]` ends (debounced) while `cup_tray`.
- **release → retreat:** wrist distance from cup increases by > 0.03 m, or 0.3 s pass.
- **any → recover:** after a **drop** event (cup unsupported, untouched for ≥
  debounce and falling ≥ `drop_min_fall`) or the cup touching the table again after
  lift-off. From recover, a new debounced `touch` goes back to **grasp** (a regrasp is
  counted). Phases may repeat; segments are a list, not one of each.

**Events** (list of dicts with `t`, `sample`, `kind`, `side`): `first_touch`,
`grasp_opposition_start/end`, `lift_off`, `tray_contact`, `release`, `drop`,
`tip_over` (cup tilt > 60°), `regrasp`, `hand_hand_contact`.

### 4.4 Outcome for `task == "cup_to_tray"`

Evaluate over the final `settle_window` seconds; every criterion is reported
individually:

| criterion | rule |
|---|---|
| `in_tray` | cup xy inside the tray's inner rim bounds (half-extents minus rim thickness) **and** `cup_tray` true, `cup_table` false |
| `upright` | angle between cup local +z and world +z < `upright_max_deg` |
| `released` | no `touch` on either side during the window |
| `settled` | cup linear/angular speed below thresholds during the whole window |
| `lifted_at_least_once` | any `lifted` sample |

`success = all(...)`. `failure_reason` is the first failing item in this order:
`never_touched, never_lifted, dropped_outside_tray, tipped_over, not_in_tray,
not_released, not_settled, too_short` (episode < 1 s or < settle_window). Unknown
tasks get `outcome = "not_evaluated"` and phases only.

### 4.5 Confidence (heuristic, document it as such)

Per segment, `confidence = clip(fresh_frac × stability × margin, 0, 1)`:
- `fresh_frac`: fraction of the segment's samples where the active side's input was
  fresh (1.0 for keyboard/scripted sources).
- `stability`: `1 / (1 + flickers)`, where flickers = raw (pre-debounce) toggles of the
  signal that started/ended the segment within ±0.2 s of its boundaries.
- `margin`: for threshold-based boundaries, `min(1, |signal − threshold| / threshold)`
  averaged over the 0.1 s after the boundary; 1.0 for contact-based boundaries.

Outcome confidence = min over the criteria's margin terms, × `fresh_frac` of the
settle window. Store all three factors per segment, not only the product, so the team
can recalibrate later.

### 4.6 Outputs

Written back into the same `episode.h5` under `labels/` (replace if present) and into
`episode.json["labels"]`:

```
labels/phase           (N,) int8, attr names
labels/segments        JSON attr: [{phase, side, start_t, end_t, start_sample, end_sample,
                                   confidence, fresh_frac, stability, margin}]
labels/events          JSON attr
labels/outcome         JSON attr: {success, failure_reason, criteria:{...}, confidence,
                                   active_side, bimanual, regrasp_count}
labels attrs           labeler_version, config (JSON of LabelConfig), created_utc
```

### 4.7 CLIs

- `scripts/label_episode.py PATH [PATH ...] [--dry-run]`: label one or more
  `episode.h5` files (or episode dirs), print a one-line summary each, and write
  results unless `--dry-run`.
- `scripts/summarize_episodes.py [DIR] [--csv out.csv]`: table of every episode under
  `data/episodes`: id, operator, source, duration, success, failure_reason, outcome
  confidence, time in each phase, regrasps, dropped video frames. This is what the team
  uses to pick good/bad demos.
- The recorder calls `label_episode` right after saving (unless `--no-auto-label`).
  A labeler exception must never lose the recording: catch it, log it, and keep
  `labels: null`.

## 5. Tests (all must run without a headset or network)

New test files:
- `test_episode_recorder.py`
  - Headless round trip: build the workspace model, run keyboard-style scripted
    targets (reuse `demo_targets`) for ~2 s with `source="scripted_demo"` and
    `video=False`. Assert shapes/dtypes match §3.4, time is strictly increasing,
    metadata keys exist, `load_demo` + `validate_model` accept it, and `replay_demo.py
    --headless` succeeds on it.
  - Video variant: `video=True` with a tiny size (160×120) and 10 fps. Skip with
    `unittest.skip` if `mujoco.Renderer` cannot be created. Assert the MP4 exists and
    the frame count equals `len(frame_time)`, which is ≈ duration × fps minus drops.
  - Fake bridge: a fake streamer class feeding `synthetic_skeleton()` plus a valid
    wrist transform into `TrackingBridge(with_wrist=True)`. Assert the new tuple
    fields 4–6 and that the recorder stores the raw skeleton bit-identically (float32
    tolerance).
  - Privacy: the address string passed to the bridge appears nowhere in the `.h5` or
    `.json` bytes; no absolute home path in metadata.
  - Discard leaves no directory. `H` path saves with `end_reason="reset"`.
- `test_episode_labels.py`: synthetic arrays only (no MuJoCo):
  - clean success (reach→grasp→lift→transport→place→release→retreat; `success=True`);
  - drop mid-transport outside the tray → `recover` phase, `dropped_outside_tray`;
  - placed but still held → `not_released`;
  - tipped in tray → `tipped_over`;
  - never touched → `never_touched`;
  - contact flicker within debounce does not create extra segments, and lowers
    `stability`;
  - stale input lowers `fresh_frac`/confidence;
  - regrasp after drop increments `regrasp_count`.
- `test_workspace_scene.py`: add the geometry-unchanged assertion from §4.1.
- Update `test_control_panel.py` for the new keys/buttons and live allow-list.

## 6. Docs

- README: new section "Recording episodes", covering keys, flags, folder layout, the
  schema table (short form), video caveats, and how to label/summarize/replay.
- TEAM_SETUP.md §6: replace the "No live arm-demo recorder" bullet with an accurate one.
  Example: a recorder now exists. Labels are heuristic, and keyboard and live episodes
  share a schema but are not equivalent demonstrations. Keep the "no trained policy"
  statement.
- Every docstring/print must stay honest: recorded episodes are **simulated robot
  state driven by human tracking**, not real-robot data.

## 7. Acceptance criteria

1. `./run_human_arms.command`: E starts, E stops, and an episode folder appears with
   `episode.h5`, `episode.json`, `cam_operator.mp4`, `cam_overview.mp4`. The status
   shows the outcome.
2. Same in live mode (`./run_live_human_arms.command`), with `input/*` populated and
   `fresh` reflecting real tracking dropouts.
3. `replay_demo.py data/episodes/<id>/episode.h5` replays the arms, hands and cup, and
   prints phase transitions.
4. `summarize_episodes.py` lists all episodes. `label_episode.py` re-labels them
   idempotently (same output twice).
5. Control-loop period with recording+video on stays within ~15 ms on an M-series Mac,
   or the PR documents the measured numbers and the reduced defaults.
6. All tests pass. No changes to replay behaviour for NPZ, flat HDF5 or HolosCapture
   files.

## 8. Out of scope (don't build)

Policy training; dex-retargeting; a movable base; real-robot output; recording audio
or the headset's own cameras (Tracking Streamer doesn't provide them here); conversion
to Holos' internal format (pending repo access, see §9); a web UI.

## 9. Open questions to leave as TODOs in the PR description

- Align `input/*` naming with HolosCapture (`data/<demo>/obs/hand_*_joints_world`)
  once Holos repo access is granted? World-frame joints are derivable as
  `wrist @ finger_skeleton`.
- Should success thresholds be tuned against a few hand-labelled episodes? (Nick will
  record ~10–20 this week: mixed successes, drops, and not-released cases.)
- Is a wrist-mounted camera view worth adding for video-model work?

## 10. Suggested commit sequence

1. `workspace_scene` constants refactor + geometry test
2. `TrackingBridge` raw fields + tests
3. `episode_recorder.py` (state only) + hooks + control panel keys + tests
4. Video rendering/encoding + tests
5. Replay support for episodes
6. `episode_labels.py` + `label_episode.py` + `summarize_episodes.py` + tests
7. README / TEAM_SETUP updates

Commit messages: imperative, scoped (`recorder: ...`, `labels: ...`).
