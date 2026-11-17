# Spec: Stream the simulation view into the Vision Pro (`--headset-view`)

**Owner:** Nick Tiesi · **Branch:** `Nick` (continue on it; don't push)
**Goal:** while teleoperating, the operator sees the MuJoCo scene in a floating 2D video
window in the headset, instead of having to look at the Mac screen.

## 0. Background (verified in the installed `avp_stream` 2.51)

- `VisionProStreamer.configure_video(device=None, size="WxH", fps=N, stereo=False)` sets up
  an outbound WebRTC video track with *synthetic* frames. Call it before `start_webrtc()`.
- `streamer.start_webrtc(port=9999, blocking=False)` starts outbound media. It works for both
  local IP and room-code (cross-network) modes. Hand tracking keeps flowing as before.
- `streamer.update_frame(frame)` takes a **BGR24** `uint8` array of shape `(H, W, 3)` that
  matches the configured size. It only stores a reference (`self.user_frame = frame`), so it
  must always be given a **new** array, never a buffer that is later mutated in place.
- Read `avp_stream/streamer.py` in `.venv` (around `configure_video`, `update_frame`,
  `start_webrtc`, `serve`) and the upstream examples it references
  (`01_visualize_hand_direct.py`) before implementing. Do **not** use `configure_mujoco`
  (the AR mode). It switches hand tracking to the sim frame and needs a USDZ converter.
  That is out of scope.

## 1. Behaviour

New flags on `view_dual_arms.py` (live mode only; error if used without `--address`):

| Flag | Default | Meaning |
|---|---|---|
| `--headset-view` | off | Stream the sim view to the headset |
| `--headset-size` | `1280x720` | Streamed frame size (even numbers) |
| `--headset-fps` | 30 | Target stream rate (wall-clock) |
| `--headset-camera` | `follow` | `follow` = whatever the Mac viewer camera currently shows (mouse/B changes it); `operator` / `overview` = the fixed workspace views from `configure_camera(..., workspace=True)` |

- Use a new `--headset-view` flag rather than changing default behaviour. The `.command`
  launchers already forward `"$@"`, so
  `./run_live_human_arms.command --auto-recenter --headset-view` works without launcher changes.
- Draw a small status strip on the streamed frame, bottom-left, in plain text: the same
  per-side `LIVE/WAIT · FOLLOW/press C` summary, `● REC mm:ss` while recording, and the last
  saved outcome. The operator can then calibrate and record without looking at the Mac. Use
  `cv2` (already installed via `avp_stream`) for the text, keeping it small and legible.
- If WebRTC fails to start or the headset never connects, print one clear line and keep
  teleoperation and recording working. The headset view must **never** crash or stall
  the control loop.

## 2. Implementation

### 2.1 New module `scripts/headset_view.py`

```python
class HeadsetView:
    def __init__(self, model, size=(1280, 720), fps=30, camera="follow"): ...
    def attach(self, streamer) -> None   # configure_video(...) + start_webrtc(blocking=False)
    def maybe_send(self, data, viewer_cam, overlay_lines) -> None  # main thread, rate-limited
    def close(self) -> None
    status: str          # "off" | "waiting for headset" | "streaming" | "error: ..."
```

- **Rendering** happens on the main thread inside the existing `viewer.lock()` block, with
  one `mujoco.Renderer` created once. Set `model.vis.global_.offwidth/offheight` ≥ size
  **before** creating any renderer (the recorder already does this for its size, so take
  the max of both).
- **Rate limit by wall clock** (`time.monotonic()`), not sim time, because the headset
  needs frames even while paused. Skip a frame when the previous render plus overlay took
  longer than the budget.
- Convert RGB → BGR (`frame[..., ::-1].copy()`), draw the overlay, then `update_frame()`.
- `follow` camera: copy `lookat`, `distance`, `azimuth`, `elevation` (and `type`/`trackbodyid`
  if not free) from `viewer.cam` into the view's own `MjvCamera` each frame. Never pass
  `viewer.cam` itself to the renderer.

### 2.2 Wiring the streamer

`TrackingBridge` currently constructs the `VisionProStreamer` inside its background
thread (`self.factory(ip=..., ...)`). Add an optional `on_streamer` callback to
`TrackingBridge.__init__`. After the streamer is constructed, call it once (inside a
try/except that stores and prints errors). `HeadsetView.attach` is passed as that
callback. Check whether `configure_video`/`start_webrtc` are safe to call from that
thread. If the library requires the main thread (signal handlers, asyncio loop), defer
the attach to the main loop: the bridge exposes `streamer_ready` and the main loop calls
`attach` once. Document which one you chose and why.

Keep `TrackingBridge` behaviour identical when `on_streamer` is None. The existing
tuple layout and tests must not change.

### 2.3 Sharing work with the recorder

When an episode is recording **and** the headset camera is `operator`, reuse the
recorder's rendered operator frame instead of rendering twice, if the frame is fresh
(rendered within the last 1/fps s). Otherwise render separately. Measure and report
the loop period with recording + headset view both on.

## 3. Performance acceptance

On Nick's Mac, with `--headset-view` and recording at 30 fps:
- Control loop median ≤ 15 ms. Report median/p95/max using the episode's `time/wall`
  diffs, as before.
- If over budget, change the **defaults** (first `--headset-size 960x540`, then
  `--headset-fps 20`) and say so in the README. Don't touch the recorder's defaults.

## 4. Tests (no headset, no network)

`test_headset_view.py`:
- A fake streamer that records `configure_video`/`start_webrtc`/`update_frame` calls.
  Assert the size/fps passed, that frames are BGR `uint8` of the right shape, that each
  `update_frame` gets a new array object, and that the rate limit holds (calling
  `maybe_send` 100× in a tight loop sends about 0–2 frames).
- `follow` copies the viewer camera's parameters. Changing them changes the next frame's
  camera, and the viewer's camera object is never mutated.
- Errors in `attach` or `update_frame` set `status="error: ..."` and do not raise.
- `TrackingBridge(on_streamer=...)` calls the callback exactly once with the fake streamer.
  With `on_streamer=None`, existing tests still pass unchanged.
- Skip the rendering-dependent assertions if a `mujoco.Renderer` can't be created (as the
  recorder's video test does).
- CLI: `--headset-view` without `--address` errors out.

## 5. Docs

README "Live recording tips": how to turn the headset view on, what the overlay shows, and
how to adjust the view by moving the Mac viewer camera. State plainly that this is a 2D video
window, not AR, and that latency depends on Wi-Fi.

## 6. Commits

1. `bridge: optional on_streamer hook` + tests
2. `headset: stream rendered sim view to Vision Pro (--headset-view)` + tests
3. `headset: reuse recorder frames and overlay status` + measured timings
4. `docs: headset view`

Final summary for Nick: the commands to run, the measured loop timings, which thread the
attach runs on, and anything that needs checking with the headset on.
