# Spec: lab3l Studio, a local web UI for recording and review

**Owner:** Nick Tiesi · **Branch:** `Nick` (continue on it; don't push)
**Goal:** run a full session without the terminal. Double-click one launcher, and a browser page
lets you start a live session, calibrate and record, then review, correct, junk and export episodes.
This is also the first version of the review frontend in the team plan (Sprints 9–10: "review
confidence and failure segments, correct labels, and export").

## 0. Decisions already made

- **Local web app.** A Python server bound to `127.0.0.1` serves a browser UI. It's for one
  operator on the Mac running the sim, not a hosted service.
- **Backend: Flask** (add `Flask>=3,<4` to `requirements.txt`). `send_file(conditional=True)`
  gives HTTP Range support, so MP4s seek properly.
- **Frontend: plain HTML/CSS/ES-module JavaScript, no build step, no CDN** (it must work offline
  in the lab). Keep it small and split into modules (`api.js`, `live.js`, `review.js`,
  `timeline.js`) so it can be ported to React later.
- **The sim stays a separate `mjpython` process** (macOS requires the MuJoCo viewer on its own main
  thread). Studio starts it and talks to it over a localhost control channel (§2). The existing
  Hand controls window keeps working alongside.

## 1. Layout

```
scripts/studio/
  server.py          # Flask app factory + CLI (python -m / run_studio.command)
  session.py         # starts/stops the sim subprocess, control-channel client, status cache
  episodes.py        # list/read episodes, review.json, junk/restore, relabel, export
  static/index.html, static/app.css, static/*.js
scripts/sim_control.py   # sim-side control server (imported by view_dual_arms.py)
run_studio.command       # double-click: starts server, opens http://127.0.0.1:8765
```

Run: `./run_studio.command` (macOS) or `.venv/bin/python -m studio.server --port 8765` from
`scripts/`. The server uses the regular venv `python`; it launches `.venv/bin/mjpython` for the sim
and for MuJoCo replays.

## 2. Sim control channel (`sim_control.py` + small hooks in `view_dual_arms.py`)

- New flag `--control-port N` (default off). When set, the sim starts a `ControlServer` thread on
  `127.0.0.1:N` (refuse any other bind address). It accepts **one** client and speaks JSON lines.
- **Client → sim:** `{"token": T, "key": "C"}`, using the same keys as the controls window
  (`C E X H B space`, plus `1 2`). Map each to the GLFW code and put it on the **existing**
  `events` queue, so the behaviour is identical to the Tk buttons. Ignore messages whose token
  doesn't match the `--control-token` given at launch, and ignore unknown keys.
- **Sim → client**, at most 10 Hz and on change: `{"status": {...}}` with per-side
  `live/follow/forearm/reach_limit`, `paused`, `recording`, `rec_seconds`, `episode_counter`, the
  recorder notice, `headset_view` status, and `last_saved: {id, folder, outcome, failure_reason,
  confidence}` when an episode has just been saved. Build it from state the loop already has; don't
  parse terminal text.
- The channel must never block the control loop. Use a thread plus queues; drop status updates
  if the client is slow. If the client disconnects, the sim keeps running.
- Without `--control-port`, behaviour is unchanged.

## 3. Live page

- **Connection form:** Tracking Streamer IP or room code. Keep it in browser `localStorage` only;
  **never** write it to disk, logs or episode metadata (TEAM_SETUP privacy rule). Also an operator
  tag, and toggles (with short help text) for auto-recenter (default on), headset view (default
  on), headset camera (follow/operator/overview), grasp assist (default off), video on/off and
  video fps.
- **Start** launches `mjpython scripts/view_dual_arms.py --wrist human --workspace --address …
  --control-port … --control-token …` plus the chosen flags, with `MENAGERIE_CACHE_DIR` set the way
  the `.command` launchers set it. Generate the token per launch. Only one session at a time;
  **Stop** sends Esc (key 256) so the sim saves any open episode, then waits, then terminates.
- **Session panel:** big status for each hand (LIVE/WAIT, FOLLOW/press C, REACH LIMIT); a REC
  timer; big buttons **Reset (H)**, **Calibrate (C)**, **Record/Stop (E)**, **Discard (X)**,
  **Pause**, **Switch view (B)**; a reminder "H → C → E". Keyboard shortcuts work while the page has
  focus. When `last_saved` arrives, show a toast "Saved ep 0024 · SUCCESS · Review →".
- Show the sim's stdout/stderr in a collapsible log panel. Keep the last ~500 lines in memory only,
  scrubbed of the address (reuse `episode_recorder.scrub` or equivalent).
- If the sim exits or crashes, show the exit code and the last log lines, and re-enable Start.

## 4. Review page

**Episode list** (left): every episode under `data/episodes/` (not `_junk`). Columns: number, time,
duration, auto outcome + reason, confidence, human status (kept/junk/unreviewed) and human outcome,
assist flag. Filters: success/fail/unreviewed/assisted. A toggle shows `_junk` episodes too.

**Episode detail** (right):
- **Video:** operator/overview tabs (`cam_*.mp4`, served with Range). Use playback speed controls
  (0.25×–2×) and frame-step buttons.
- **Timeline under the video:** coloured phase segments from `labels.segments`, event ticks from
  `labels.events` (first_touch, lift_off, tray_contact, release, drop_in, drop, tip_over, regrasp,
  post_release_touch), and a playhead. Clicking seeks the video. Map video time ↔ sim time with
  `video/<camera>/frame_time` (frame k ↔ `frame_time[k]`), not `currentTime` alone, because frames
  may be dropped.
- **Labels panel:** auto outcome, criteria with pass/fail, margins and confidence, placement
  (placed/dropped_in), regrasps, and the segment table with confidence factors.
- **Actions:** **Replay in MuJoCo** (launch `replay_demo.py` with mjpython; disabled while a live
  session runs, or warn about performance), **Re-label** (re-run `label_episode` on this episode),
  **Junk / Restore** (move the folder into or out of `data/episodes/_junk/`, with a confirm step;
  never delete), and a free-text **Note**.

## 5. Human corrections (`review.json`, one per episode folder)

```json
{"schema": "lab3l.review", "version": 1, "reviewer": "NT", "updated_utc": "...",
 "status": "kept | junk | unreviewed", "note": "...",
 "outcome": {"success": true, "failure_reason": null} | null,
 "segments": [{"phase": "...", "start_t": 0.0, "end_t": 1.2}] | null,
 "auto_labeler_version": "2"}
```
- **Never overwrite auto labels.** Corrections live only in `review.json`. Re-labeling never touches
  `review.json`; show a "labeler version changed since review" badge when they differ.
- **Outcome override:** a success/fail switch plus a failure-reason dropdown (`FAILURE_ORDER`).
- **Phase correction:** "Edit phases" makes the timeline boundaries draggable. Snap to sample
  times, keep segments contiguous and non-overlapping, and allow changing a segment's phase from a
  dropdown, split at the playhead, and merge with the next segment. Save writes `segments`; "Reset
  to auto" clears it.
- `status` is set by Keep/Junk. Junking also moves the folder; restoring sets `kept`.
- Write atomically (temp file + rename).

## 6. Summary and agreement

- `summarize_episodes.py` gains columns `human_status`, `human_success`, `human_reason`,
  `agrees_with_auto`. It works unchanged when no `review.json` exists.
- The review page header shows: N episodes, N reviewed, auto-vs-human outcome agreement (x/y), and
  a small confusion count (auto success/fail × human success/fail). That measures the labeler.

## 7. Export

**Export dataset** builds `data/exports/lab3l_<YYYYmmdd-HHMM>.zip` containing every **kept**
episode (`episode.h5`, `episode.json`, `review.json`, optionally the MP4s, with a checkbox; the
default is to include them), `summary.csv` (with the human columns), and a `README.txt` (schema
versions, labeler version, scene version, how to load, and a note that episodes are simulated
robot state driven by human tracking). Stream it to disk, show progress, then offer a download
link. Don't build the zip in memory. Exclude `_junk` and exports.

## 8. Security and robustness (it binds to localhost, but still)

- Bind `127.0.0.1` only. Reject requests whose `Host` isn't `127.0.0.1:<port>` or `localhost:<port>`
  (DNS-rebinding guard). Every state-changing request (POST/PUT/DELETE) must carry an
  `X-Studio-Token` header that matches a token generated at server start and injected into the
  page.
- Resolve every episode path inside `data/episodes` (or `_junk`). Reject `..`, absolute paths and
  symlinks that escape.
- Subprocesses use argument lists, never `shell=True`. Pass the address as one argument.
- Clean up: on server shutdown (Ctrl-C or the window closing), stop any running sim with Esc
  first, then kill it if it doesn't exit.

## 9. Tests (no headset, no network, no display)

- `test_sim_control.py`: a `ControlServer` with a real socket. Token required; keys map to the
  right GLFW codes on a queue; status messages are rate-limited; a slow or disconnected client never
  blocks `publish()`; it refuses non-loopback binds.
- `test_studio_api.py` (Flask test client, temp `data/` with 2–3 small episodes produced by the
  headless recorder or synthetic files):
  - listing and filters; detail JSON includes segments, events and frame_time;
  - MP4 Range requests return 206;
  - the token and Host checks reject bad requests;
  - path traversal is rejected;
  - review.json round-trip (outcome override, segment edits validated contiguous);
  - junk/restore move the folders;
  - re-label leaves review.json untouched;
  - export zip contents (kept only, CSV human columns).
- `test_studio_session.py`: a fake sim subprocess (a small Python script speaking the control
  protocol). Start, status flow, send key, Stop sends Esc, crash shows its exit code, only one
  session at a time, and the address never appears in logs or on disk.
- Update `test_summarize` coverage for the new columns. The full suite must pass.

## 10. Docs

README section "lab3l Studio": how to launch it, a screenshot list of the pages (text is fine),
the review workflow (watch → check labels → keep/junk → correct → export), what review.json is,
and the privacy note about the IP. Add Studio to TEAM_SETUP as the recommended way to record.

## 11. Commits

1. `sim: localhost control channel (--control-port)` + tests
2. `studio: server, episode API, review.json` + tests
3. `studio: review page (video, timeline, labels, junk/notes)`
4. `studio: phase/outcome correction + agreement stats`
5. `studio: live page (launch, controls, status, log)` + session tests
6. `studio: dataset export` + tests
7. `docs: lab3l Studio`

Final summary for Nick: how to launch it, what to click through to verify on his Mac (live session
with the headset; reviewing episode 0016; junk and restore one; correct one phase; export), and
anything not testable headless.
