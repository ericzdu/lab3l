# Spec: Make cup-to-tray reachable + optional auto-recenter

**Owner:** Nick Tiesi · **Branch:** `Nick` (continue on it; don't push) · **Follows:**
`SPEC_episode_recorder_and_labels.md` (already implemented).

## 0. Problem (measured)

From the first live episode (`data/episodes/20261026-164340_cup_to_tray_0001`) and a
headless probe of `build_human_model(workspace=True)`:

- The arms *do* follow tracking (robot wrists moved 8–30 cm), but motion is confined.
  `HumanArmController.set_target` clips the wrist to **±16 cm (x), −16/+18 cm (z)** around
  `home` (right home ≈ `[0.196, -0.25, 0.753]`), and then to **0.53 m from the shoulder**.
  At `--scale 0.6`, that box is only about ±27 cm of operator hand travel. During the
  episode the right arm was at REACH LIMIT **75 %** of the time (left 23 %).
- **The cup sits at the bottom edge of that box.** Cup centre `[0.32, -0.28, 0.562]`,
  top rim z ≈ 0.613. The lowest achievable right wrist is z ≈ 0.62–0.63 (with
  `dz = -0.16`). There the palm only *rests on the cup's rim* (hand–cup gap −0.5 to
  −0.7 cm), with no room to wrap fingers around the cup body. The tray centre
  (x = 0.43) is also near or beyond the forward limit.
- Any tracking drop longer than 0.5 s calls `WristMapper.invalidate(side)`, and the arm
  then **freezes until C is pressed** (Emma's deliberate no-jump safety rule). Wi-Fi
  blips are frequent, so from the operator's view the "arms stop moving".

Net effect: live cup-to-tray success is close to impossible. The scene geometry, not the
operator, is the bottleneck.

## 1. Ground rules

- Continue on branch `Nick`. Read the files you touch first. `human_arms.py`,
  `arm_control.py`, `spherical_wrist.py` and `workspace_scene.py` belong to Emma's
  baseline: change only what this spec asks, and list every behavioural change in the
  final summary so Nick can tell her.
- **Don't change limb lengths, the clip box, the 0.53 m reach cap, or the
  `WristMapper` default behaviour.** Move the *props/table*, not the robot.
- The Panda and spherical-wrist variants (`arm_control.py`, `spherical_wrist.py`) must
  keep their current table geometry and behaviour.
- Old recorded episodes must still label correctly (see §3.3).
- Full suite must pass before each commit:
  `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`.

## 2. Task A — Reachability probe (`scripts/check_reach.py`)

A headless CLI (no viewer, no headset) that builds the workspace model and, for each
side, does the following:
1. Sweeps wrist targets over a grid (dx, dy, dz in the clip box, 2 cm steps) at
   three wrist orientations: palm-down (home), palm-inward (rotate ±90° about the
   forearm, as U/P does), and palm-inward + 30° pitch down.
2. Settles physics (~3 s sim) for each target and records the achieved wrist position,
   `reach_errors`, and the min signed distance between hand geoms and cup geoms
   (`mujoco.mj_geomDistance`), plus the same against the tray interior.
3. Prints a compact report: the best "wrap" pose for the cup, meaning a palm-inward pose
   where thumb and at least one finger group are within 1 cm of the cup wall at its
   mid-height ± 2 cm. Also report whether a cup held in that pose could be lowered into
   the tray's inner bounds.

`--json out.json` writes the full grid. Reuse `setup()` from `view_dual_arms.py`. Use
the model built from the *current* constants, so the probe can be re-run after §3.

## 3. Task B — Move the props into the workspace

### 3.1 Geometry change (human-arm scene only)

- Make the human-arm worktop height come from `workspace_scene.TABLE_TOP`, removing the
  hard-coded `.48` in `human_arms.py` (worktop centre = `TABLE_TOP - 0.025`). Leave
  `arm_control.py` / `spherical_wrist.py` untouched.
- Pick new values for **`TABLE_TOP`, `CUP_START` (x, y) and `TRAY_CENTER`** using the
  probe, under these constraints:
  - The right hand can reach a palm-inward wrap of the cup at mid-height with **≥ 3 cm
    clearance from every clip/reach limit** (no REACH LIMIT in that pose).
  - The right hand, holding the cup, can place it inside the tray's inner bounds
    with ≥ 3 cm margin from the limits.
  - The cup starts upright and at rest, at least 6 cm from the tray rim, and does not
    collide with either arm's home pose.
  - Keep the cup on the right-hand side and the tray between the hands, as now.
  - Raising the table (and with it the cup and tray) is the preferred lever, because
    the cup is too low. Shift x/y only as needed. Table legs must still reach the floor
    (adjust their height/centre from `TABLE_TOP`).
- Update `test_workspace_scene.py`: it currently asserts the geometry is *unchanged*.
  Change it to assert the new constants and that the geometry follows them.

### 3.2 Scripted proof: `scripts/scripted_cup_to_tray.py`

A deterministic, headless (optional `--viewer`) pick-and-place for the **right** hand,
built from controller targets plus `FingerTransition`:
1. Pre-grasp above/beside the cup, palm inward.
2. Descend to the wrap pose.
3. Close.
4. Lift ≥ 5 cm.
5. Move over the tray.
6. Lower until the cup touches the tray.
7. Open slowly.
8. Retreat.

It records the run through `EpisodeRecorder` (`source="scripted_demo"`,
`--no-video` by default) and prints the labeler's outcome.

- **Acceptance:** the labeler scores it **SUCCESS** with the default `LabelConfig`, and
  the phase sequence includes reach → grasp → lift → transport → place → release.
- Add a test (`test_scripted_cup_to_tray.py`, video off) that runs it and asserts
  success. If the test takes longer than ~20 s, mark it so it can be skipped with an
  env var (`LAB3L_SKIP_SLOW=1`), but keep it running by default.
- If no geometry satisfying §3.1 gives a scripted success, **stop and report**: what you
  tried, the probe numbers, and why it fails. Do **not** loosen the labeler thresholds
  or the controller limits to force a pass.

### 3.3 Keep old episodes valid

- `episode_labels.py` must take `table_top`, `cup_start` and `tray_center` from the
  episode's `metadata.scene`, which the recorder already stores, and fall back to
  module constants only if they're missing. Check this is true everywhere the labeler
  uses scene geometry, including the cup's rest height and the tray bounds.
- Add `"scene_version": 2` to the recorder's `metadata.scene`, and treat a missing
  value as version 1.
- Add a test: an episode created with the old constants (synthetic arrays + old
  metadata) labels identically before and after the change.
- `replay_demo.py` rebuilds the model from current code. If the episode's
  `scene.table_top`/`cup_start`/`tray_center` differ from the current constants, print
  a clear warning that the props in replay are at their *new* positions. The robot
  replay is still valid because nq/nv/nu are unchanged.

## 4. Task C — Optional `--auto-recenter` (live only, default **off**)

The current rule (freeze after any blip until C) stays the default. With
`--auto-recenter`:
- When a side's tracking becomes fresh again within `--recenter-window` seconds
  (default 3.0) after it was lost, and that side was calibrated before the loss, call
  the same per-side calibration that **C** performs, anchored at the robot's *current*
  wrist pose. No jump is possible, because the mapping is relative.
- Preserve that side's previous `position_scales` (including the hand-span lateral
  gain) instead of resetting it to `scale`. Add a `WristMapper.recenter(side, ...)`
  method rather than changing `calibrate()`'s behaviour.
- Losses longer than the window still need C, as now.
- Log an `auto_recenter` event (side, gap seconds) when recording. The status shows
  `arm FOLLOW (auto)` for one second afterwards.
- Tests: synthetic fresh/stale sequences via the mapper, with no headset. A short gap
  re-anchors with no target jump; a long gap stays frozen; a side never calibrated is
  never auto-calibrated.

## 5. Docs

- README "Recording episodes": a short "Tips for live recording" list: press H then C
  before each episode, keep hands within about 25 cm of the calibration pose, watch for
  REACH LIMIT, press C after any `arm press C`, and use `--auto-recenter` on flaky Wi-Fi.
- README/TEAM_SETUP: document the new table/cup/tray positions and `check_reach.py`.

## 6. Commits

1. `scene: drive human-arm worktop from TABLE_TOP` (no value change yet) + tests
2. `tools: add headless reachability probe`
3. `labels: read scene geometry from episode metadata` + old-episode test
4. `scene: move table, cup and tray into the arms' workspace` + updated tests
5. `demo: scripted right-hand cup-to-tray that labels as success` + test
6. `live: optional --auto-recenter after short tracking gaps` + tests
7. `docs: live recording tips, new scene layout, reach probe`

Final summary for Nick: the chosen constants with probe numbers before and after,
the scripted episode's labels, every change to Emma's files, and anything you could
not verify without the headset.
