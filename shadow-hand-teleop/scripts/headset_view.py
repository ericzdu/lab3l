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


class HeadsetView:
    def __init__(self, model, size=(1280, 720), fps=30, camera="follow", budget=.012,
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
        self.sent = self.skipped = 0
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

    def maybe_send(self, data, viewer_cam, overlay_lines=()):
        """Render and send one frame if one is due; never raises."""
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
            rgb = self.render(data, viewer_cam)
            # New array every frame: update_frame only stores a reference.
            frame = np.ascontiguousarray(rgb[..., ::-1])
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
        return {"status": self.status, "sent": self.sent, "skipped": self.skipped,
                "render_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                              "max": float(ms.max())} if len(ms) else None}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
