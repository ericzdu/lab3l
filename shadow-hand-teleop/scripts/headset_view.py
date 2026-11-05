"""Stream the simulation view to the Vision Pro as a floating 2D video window.

Uses avp_stream's outbound WebRTC video with synthetic frames: configure_video()
then start_webrtc(), and update_frame() with a new BGR24 array per frame. This is
a flat video window, not AR (configure_mujoco is deliberately not used).

Threads: rendering happens on the caller's (main) thread, rate-limited by wall
clock so frames keep flowing while paused. attach() is called on the tracking
bridge's thread (TrackingBridge on_streamer hook). start_webrtc() needs no main
thread (it runs its own asyncio loop), but it blocks: about 0.5 s on a local IP,
and with a room code until the headset connects. So attach() runs it on its own
daemon thread and returns at once, and neither hand tracking nor the control
loop ever waits for the headset. Every failure only changes `status`.
"""
from __future__ import annotations

import threading
import time

import mujoco
import numpy as np

CAMERAS = ("follow", "operator", "overview")
CONNECT_HINT_SECONDS = 30.   # print one hint if no headset connects by then
REC_MARK = "\u25cf"           # "●" in recorder status: drawn as a red dot


def plain(text):
    """cv2's Hershey fonts are ASCII-only."""
    text = text.replace("\u00b7", "|").replace("\u2013", "-").replace("\u2014", "-")
    return text.encode("ascii", "replace").decode()


def draw_overlay(frame, lines):
    """Small status strip, bottom-left, on a BGR frame (in place)."""
    import cv2
    lines = [line for line in lines if line]
    if not lines:
        return frame
    height, width = frame.shape[:2]
    scale = .5 * height / 720
    thickness = max(1, int(round(height / 720)))
    step = int(28 * height / 720)
    margin = int(12 * height / 720)
    dot = int(7 * height / 720)
    texts = []
    for line in lines:
        mark = line.startswith(REC_MARK)
        texts.append((mark, plain(line[1:].strip() if mark else line)))
    widths = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
              + (3 * dot if mark else 0) for mark, text in texts]
    top = height - margin - step * len(texts)
    right = min(width, 2 * margin + max(widths))
    # Darken the strip so white text stays legible on any background.
    frame[max(0, top - margin // 2):height, 0:right] //= 3
    for i, (mark, text) in enumerate(texts):
        baseline = top + step * (i + 1) - step // 3
        x = margin
        if mark:
            cv2.circle(frame, (x + dot, baseline - dot), dot, (40, 40, 230), -1, cv2.LINE_AA)
            x += 3 * dot
        cv2.putText(frame, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


class HeadsetView:
    def __init__(self, model, size=(960, 540), fps=20, camera="follow", budget=.016,
                 clock=time.monotonic):
        width, height = (int(v) for v in size)
        if width % 2 or height % 2 or width < 16 or height < 16:
            raise ValueError("headset size must be even, at least 16 pixels per side")
        if not np.isfinite(fps) or not 0 < fps <= 60:
            raise ValueError("headset fps must be in (0, 60]")
        if camera not in CAMERAS:
            raise ValueError(f"headset camera must be one of {CAMERAS}")
        self.model, self.size, self.fps, self.camera_mode = model, (width, height), float(fps), camera
        self.budget, self.clock = budget, clock
        # Must happen before any renderer for this model exists; the recorder
        # does the same for its own size, so both take the max.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.camera = mujoco.MjvCamera()
        if camera != "follow":
            from view_dual_arms import configure_camera
            configure_camera(self.camera, overview=camera == "overview", workspace=True)
        self.renderer = None
        self.streamer = None
        self.status = "off"
        self.messages = []
        self.next_time = 0.
        self.attached_at = None
        self.hinted = False
        self.sent = self.skipped = self.reused = 0
        self.render_seconds = []

    # -- streamer side (bridge thread) --------------------------------------
    def attach(self, streamer):
        """configure_video now, start_webrtc on a daemon thread; never raises."""
        try:
            width, height = self.size
            streamer.configure_video(device=None, size=f"{width}x{height}", fps=int(round(self.fps)),
                                     stereo=False)
            self.streamer = streamer
            self._set("waiting for headset")
            self.attached_at = self.clock()
            threading.Thread(target=self._start, name="headset-webrtc", daemon=True).start()
        except Exception as exc:
            self._fail(exc)

    def _start(self):
        try:
            self.streamer.start_webrtc(blocking=False)
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc):
        from episode_recorder import scrub
        self._set(f"error: {type(exc).__name__}: {scrub(exc)}")

    def _set(self, status):
        if status != self.status:
            self.status = status
            if status.startswith("error"):
                self.messages.append(f"Headset view stopped ({status[7:]}); teleoperation continues.")
            elif status == "streaming":
                self.messages.append("Headset view: streaming to Vision Pro.")

    def pop_messages(self):
        """One-line notices for the terminal (status changes, connect hint)."""
        messages, self.messages = self.messages, []
        return messages

    # -- main thread -----------------------------------------------------------
    def connected(self):
        try:
            return bool(self.streamer.is_connected())
        except AttributeError:   # older avp_stream: assume yes once started
            return True

    def follow(self, viewer_cam):
        """Copy the Mac viewer camera's pose into our own camera (never mutate theirs)."""
        camera = self.camera
        camera.type = viewer_cam.type
        camera.trackbodyid = viewer_cam.trackbodyid
        camera.fixedcamid = viewer_cam.fixedcamid
        camera.lookat[:] = viewer_cam.lookat
        camera.distance = viewer_cam.distance
        camera.azimuth = viewer_cam.azimuth
        camera.elevation = viewer_cam.elevation
        if hasattr(viewer_cam, "orthographic"):
            camera.orthographic = viewer_cam.orthographic

    def render(self, data, viewer_cam=None):
        """RGB frame of the configured size, or None if rendering is unavailable."""
        if self.renderer is None:
            width, height = self.size
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        if self.camera_mode == "follow" and viewer_cam is not None:
            self.follow(viewer_cam)
        self.renderer.update_scene(data, camera=self.camera)
        return self.renderer.render()

    def fit(self, rgb):
        """Aspect-preserving resize of another camera's frame, letterboxed."""
        import cv2
        width, height = self.size
        h, w = rgb.shape[:2]
        scale = min(width / w, height / h)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        out = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = (width - size[0]) // 2, (height - size[1]) // 2
        out[y:y + size[1], x:x + size[0]] = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        return out

    def maybe_send(self, data, viewer_cam, overlay_lines=(), reuse=None):
        """Render and send one frame if one is due; never raises.

        reuse: (RGB frame, monotonic time) of the recorder's operator frame.
        With the 'operator' camera it replaces a render when it is at most one
        headset frame period old (letterboxed if the sizes differ).
        """
        if self.streamer is None or self.status.startswith("error"):
            return
        now = self.clock()
        if now < self.next_time:
            return
        period = 1 / self.fps
        self.next_time = now + period
        try:
            if not self.connected():
                if (not self.hinted and self.attached_at is not None
                        and now - self.attached_at > CONNECT_HINT_SECONDS):
                    self.hinted = True
                    self.messages.append(
                        f"Headset view: no headset connected after {CONNECT_HINT_SECONDS:.0f} s. "
                        "Keep Tracking Streamer open; teleoperation is unaffected.")
                self._set("waiting for headset")
                return
            self._set("streaming")
            started = time.perf_counter()
            if (self.camera_mode == "operator" and reuse is not None
                    and now - reuse[1] <= period + 1e-9):
                rgb = reuse[0] if reuse[0].shape[:2] == self.size[::-1] else self.fit(reuse[0])
                self.reused += 1
            else:
                rgb = self.render(data, viewer_cam)
            # New array every frame: update_frame only stores a reference.
            frame = np.ascontiguousarray(rgb[..., ::-1])
            draw_overlay(frame, overlay_lines)
            self.streamer.update_frame(frame)
            cost = time.perf_counter() - started
            self.render_seconds.append(cost)
            self.sent += 1
            if cost > self.budget:
                # Over budget: give the control loop the next slot back.
                self.next_time += period
                self.skipped += 1
        except Exception as exc:
            self._fail(exc)

    def stats(self):
        ms = np.asarray(self.render_seconds) * 1000
        return {"status": self.status, "sent": self.sent, "skipped": self.skipped, "reused": self.reused,
                "render_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                              "max": float(ms.max())} if len(ms) else None}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
