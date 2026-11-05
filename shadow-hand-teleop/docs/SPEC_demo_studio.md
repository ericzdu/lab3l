# Spec: Demo cut of lab3l Studio + decluttered live UI

**Owner:** Nick Tiesi · **Branch:** `Nick` (continue on it; don't push) · **Deadline:** demo in a few days.
**Goal:** a recording session that looks clean on a projector and needs no terminal. The story for
the audience: *operator records an attempt → seconds later the page shows the labeled result with
video.* Build only what serves that. `SPEC_studio_ui.md` is the post-demo backlog: build the parts
of it named below and nothing else from it.

## 1. Declutter the live UI (everyone benefits, not only Studio)

- **New `--demo` flag on `view_dual_arms.py`:**
  - launch the MuJoCo viewer with `show_left_ui=False, show_right_ui=False`;
  - hide the green target spheres (`{side}_goal` geoms), for example by setting their alpha to 0
    at runtime, without changing the model's nq/nv/nu or recorded data;
  - default to the workspace operator camera.
- **Compact Hand controls window in live mode.** It currently shows ~14 movement/wrist/fist
  buttons that are disabled in live mode. In live mode show only the status line and Calibrate (C),
  Record/Stop (E), Discard (X), Reset (H), Pause (Space) and Switch view (B), with larger buttons.
  Keyboard mode is unchanged. Update `test_control_panel.py`.
- **Quieter terminal:**
  - set `GRPC_VERBOSITY=ERROR` (and anything else that silences the `fork_posix.cc` /
    `ev_poll_posix.cc` lines) in the live launchers and in Studio's sim launch;
  - print the per-hand status line at most once per second, plus immediately on FOLLOW/press-C
    changes, recording start/stop and errors;
  - do not hide errors.
- **`--no-panel`:** don't open the Tk window (Studio uses this, since its page replaces the
  window). C/E/X/H/Space/B then come only from the control channel.

## 2. Control channel

Implement exactly as in `SPEC_studio_ui.md` §2 (`--control-port`, `--control-token`, localhost
only, JSON lines, keys feed the existing `events` queue, rate-limited status including `last_saved`,
never blocks the loop).

## 3. Studio, demo cut (one page)

Server and layout as in `SPEC_studio_ui.md` §0–§1 (Flask, plain JS, no build step,
`run_studio.command`, `127.0.0.1:8765`). Security from §8: localhost bind, Host check, POST token,
path guard, no `shell=True`. One page, designed to read well on a projector: large type, high
contrast, little text.

**A. Session (top).**
- **Connect form:** IP/room code (browser localStorage only, never on disk), operator tag, and
  toggles for auto-recenter (on), headset view (on), grasp assist (off). Start/Stop. Collapse the
  form once a session runs; show a small "Session settings" link to reopen it.
- **Launch:** `view_dual_arms.py --wrist human --workspace --demo --no-panel --control-port … --control-token …`
  plus the toggles. Stop sends Esc first.
- **Hand chips:** two large chips (Left / Right): green = LIVE + FOLLOW, amber = LIVE but needs C,
  grey = no tracking.
- **Step guide:** **① Reset → ② Calibrate → ③ Record**. Highlight the next needed step from live
  status (cup not at start → Reset; any arm "press C" → Calibrate; else Record). This doubles as
  instructions for the audience.
- **Buttons and shortcuts:** large Reset, Calibrate, Record/Stop (red with a running timer while
  recording) and Discard. Keyboard shortcuts H / C / E / X while the page has focus.

**B. Result (main area).**
- When `last_saved` arrives, show the episode automatically:
  - an outcome badge (**SUCCESS**, or **FAIL · dropped outside tray** with friendly wording for
    each `failure_reason`), confidence as a simple bar, and placement (placed / dropped in);
  - `cam_overview.mp4` autoplaying muted (operator view on a tab), with a **phase timeline** under
    it: coloured segments, event ticks, a playhead, and click to seek. Use
    `video/<camera>/frame_time` to map video time to sim time (`SPEC_studio_ui.md` §4).
- Below the video: a one-line phase strip, e.g. "reach 1.1 s · grasp 1.1 s · lift · transport
  3.8 s · place 1.9 s · release · retreat".

**C. Episodes strip (bottom or side).**
- Recent kept episodes (`data/episodes/`, not `_junk`) as small cards with number + outcome badge.
  Clicking one loads it into Result. This is also the demo fallback: if a live grasp fails on
  stage, click a good earlier episode (e.g. 0016).
- A tiny header count: "14 episodes · 7 successes".

**D. Collapsed log.** A "Show log" toggle with the last ~200 lines of sim output, kept in memory
only and scrubbed of the address. Shown automatically, with the exit code, if the sim crashes.

**Not in the demo cut** (stay in `SPEC_studio_ui.md` for later): review.json corrections, phase
dragging, junk/restore, notes, agreement stats, export, Replay-in-MuJoCo and Re-label buttons, and
filters.

## 4. Tests (no headset, no display)

- `test_sim_control.py`, as in `SPEC_studio_ui.md` §9.
- `test_studio_api.py`, cut down:
  - episode list and detail JSON (segments, events, frame_time, friendly reason);
  - MP4 Range returns 206;
  - token, Host and path-traversal rejection.
- `test_studio_session.py`, with a fake sim script:
  - start, status flow and keys;
  - Stop sends Esc; crash shows its exit code;
  - only one session at a time;
  - the address never appears in logs or on disk;
  - `--demo --no-panel` flags are passed.
- Unit-test the step-guide logic (a pure function from status to next step) in Python or as a tiny
  JS test runnable with Node if available. Otherwise keep the logic trivial and documented.
- `test_control_panel.py` covers the compact live layout. The full suite must pass.

## 5. Docs

- README "lab3l Studio (demo cut)": launch, the three steps, what the result card shows.
- Add `docs/DEMO_CHECKLIST.md` with:
  - **Before:** charge the headset; same Wi-Fi; close other sims; `./run_studio.command`; test one
    attempt; pick a fallback episode.
  - **During:** Start → ① ② ③ → attempt → result plays.
  - **If a grasp fails:** show the failure label (it's a feature), then click a good episode.
  - **If tracking drops:** amber chip → Calibrate.

## 6. Commits

1. `live: --demo declutter, compact live controls, quieter logs, --no-panel` + tests
2. `sim: localhost control channel` + tests
3. `studio: server + episode API (demo cut)` + tests
4. `studio: demo page (session, step guide, result video + timeline, episodes)` + session tests
5. `docs: Studio demo cut + demo checklist`

Commit 1 is useful on its own. If time runs out, stop after it and demo with the decluttered
windows. Final summary for Nick: launch steps, what to click through on his Mac with the headset,
measured loop timing with Studio connected (must stay ≈13 ms median), and anything untested.
