"""Operator/overview camera video for recorded episodes.

Frames are rendered on the caller's (main) thread with one reused
mujoco.Renderer and encoded to MP4 on a background thread. If the encoder
falls behind, frames are dropped and counted rather than slowing the control
loop. Each written frame's sim time and sample index are kept so labels can
be aligned to frames exactly, even with drops.

The two cameras are free cameras (no model change). Renders are staggered by
half a frame period so one loop iteration renders at most one camera.
"""
from __future__ import annotations

from queue import Full, Queue
import threading
import time

import mujoco
import numpy as np


CAMERAS = ("operator", "overview")


def camera_metadata(camera, model, width, height):
    fovy = float(model.vis.global_.fovy)
    focal = (height / 2) / np.tan(np.deg2rad(fovy) / 2)
    return {"lookat": camera.lookat.tolist(), "distance": float(camera.distance),
            "azimuth": float(camera.azimuth), "elevation": float(camera.elevation),
            "fovy": fovy, "width": width, "height": height,
            "fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2}


class EpisodeVideo:
    def __init__(self, model, fps=30, size=(640, 480), queue_size=60):
        from view_dual_arms import configure_camera
        width, height = size
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video fps must be positive and finite")
        if width % 2 or height % 2:
            raise ValueError("video width and height must be even (yuv420p)")
        self.model, self.fps, self.size = model, float(fps), (int(width), int(height))
        # Must happen before any renderer for this model exists.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.cameras = {}
        for name in CAMERAS:
            camera = mujoco.MjvCamera()
            configure_camera(camera, overview=name == "overview")
            self.cameras[name] = camera
        self.queue_size = queue_size
        self.renderer = None
        self.status = "ok"
        self.thread = None

    def available(self):
        """Create the shared renderer once; never raise if the platform refuses."""
        if self.renderer is None and self.status == "ok":
            try:
                width, height = self.size
                self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            except Exception as exc:  # GL/platform specific; video is optional.
                from episode_recorder import scrub
                self.status = f"unavailable: {type(exc).__name__}: {scrub(exc)}"
        return self.renderer is not None

    def begin(self, directory):
        self.directory = directory
        self.queue = Queue(maxsize=self.queue_size)
        self.dropped = {name: 0 for name in CAMERAS}
        self.frame_time = {name: [] for name in CAMERAS}
        self.sample_index = {name: [] for name in CAMERAS}
        self.render_seconds = []
        self.error = None
        period = 1 / self.fps
        self.next_time = {name: i * period / len(CAMERAS) for i, name in enumerate(CAMERAS)}
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def maybe_render(self, data, t, sample_index):
        """Render at most one due camera; t is the episode's rebased sim time."""
        if self.thread is None or self.error:
            return
        period = 1 / self.fps
        due = [name for name in CAMERAS if t + 1e-9 >= self.next_time[name]]
        if not due:
            return
        name = min(due, key=self.next_time.get)
        # Sim time can outrun the frame period (slow frames): skip, and count.
        missed = max(0, int((t - self.next_time[name] + 1e-9) // period))
        self.dropped[name] += missed
        self.next_time[name] += (missed + 1) * period
        started = time.perf_counter()
        self.renderer.update_scene(data, camera=self.cameras[name])
        frame = self.renderer.render().copy()
        self.render_seconds.append(time.perf_counter() - started)
        try:
            self.queue.put_nowait((name, frame, t, sample_index))
        except Full:
            self.dropped[name] += 1

    def _write(self):
        import imageio.v2 as imageio
        writers = {}
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                name, frame, t, sample_index = item
                if self.error:
                    self.dropped[name] += 1
                    continue
                if name not in writers:
                    writers[name] = imageio.get_writer(
                        self.directory / f"cam_{name}.mp4", format="FFMPEG", mode="I",
                        fps=self.fps, codec="libx264", quality=None, pixelformat="yuv420p",
                        macro_block_size=2, ffmpeg_log_level="error",
                        ffmpeg_params=["-crf", "23", "-preset", "veryfast"])
                writers[name].append_data(frame)
                self.frame_time[name].append(t)
                self.sample_index[name].append(sample_index)
        except Exception as exc:
            from episode_recorder import scrub
            self.error = f"error: {type(exc).__name__}: {scrub(exc)}"
            # Keep draining so the producer never blocks on a dead consumer.
            while self.queue.get() is not None:
                pass
        finally:
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    from episode_recorder import scrub
                    self.error = self.error or f"error: {type(exc).__name__}: {scrub(exc)}"

    def finish(self):
        """Flush the encoder; return (datasets, metadata) for the episode file."""
        self.queue.put(None)
        self.thread.join()
        self.thread = None
        width, height = self.size
        render_ms = np.asarray(self.render_seconds) * 1e3
        metadata = {
            "status": self.error or self.status, "fps": self.fps,
            "cameras": {name: camera_metadata(camera, self.model, width, height)
                        for name, camera in self.cameras.items()},
            "frames": {name: len(self.frame_time[name]) for name in CAMERAS},
            "dropped_frames": dict(self.dropped),
            "render_ms": {"mean": float(render_ms.mean()) if len(render_ms) else None,
                          "p95": float(np.percentile(render_ms, 95)) if len(render_ms) else None,
                          "max": float(render_ms.max()) if len(render_ms) else None},
        }
        datasets = {}
        for name in CAMERAS:
            datasets[f"video/{name}/frame_time"] = np.asarray(self.frame_time[name], dtype=np.float64)
            datasets[f"video/{name}/sample_index"] = np.asarray(self.sample_index[name], dtype=np.int64)
        return datasets, metadata

    def abort(self):
        if self.thread is not None:
            self.queue.put(None)
            self.thread.join()
            self.thread = None

    def close(self):
        self.abort()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
