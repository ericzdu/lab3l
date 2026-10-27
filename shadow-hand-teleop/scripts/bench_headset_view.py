"""Measure the control loop with the viewer open, recording video and a headset view.

No headset: frames go to a fake, always-connected streamer, so this measures
the main-thread cost (render, overlay, hand-off), not WebRTC encoding or Wi-Fi.
It mimics view_dual_arms' loop (10 ms pacing, scripted demo motion) and
records a 30 fps video episode, then reports the loop period from that
episode's time/wall, as for the recorder. Close other simulations first.

    .venv/bin/mjpython scripts/bench_headset_view.py                   # defaults
    .venv/bin/mjpython scripts/bench_headset_view.py --camera none     # baseline
    .venv/bin/mjpython scripts/bench_headset_view.py --size 1280x720 --fps 30
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time

import mujoco
import mujoco.viewer
import numpy as np

from arm_control import SIDES, WristMapper
from episode_recorder import EpisodeRecorder
from headset_view import CAMERAS, HeadsetView
from view_dual_arms import (PhysicsClock, advance_physics, configure_camera, demo_targets,
                            headset_lines, setup, video_size)


class FakeStreamer:
    def configure_video(self, **kwargs):
        pass

    def start_webrtc(self, **kwargs):
        return True

    def is_connected(self):
        return True

    def update_frame(self, frame):
        self.frame = frame


def run(camera, size, fps, seconds, record_dir, budget=.016):
    import h5py
    model, data, controller, hands = setup("human", workspace=True)
    headset = None if camera == "none" else HeadsetView(model, size=size, fps=fps, camera=camera, budget=budget)
    recorder = EpisodeRecorder(model, controller, hands, record_dir=record_dir, task="cup_to_tray",
                               source="scripted_demo", wrist_mode="human", workspace=True, scale=.6,
                               video=True, video_fps=30, video_size=(640, 480), max_seconds=3600,
                               auto_label=False)
    if headset:
        headset.attach(FakeStreamer())
        time.sleep(.2)
    mapper = WristMapper()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            configure_camera(viewer.cam, workspace=True)
        clock = PhysicsClock(model.opt.timestep)
        last = time.monotonic()
        recorder.start(data)
        begin = time.monotonic()
        while viewer.is_running() and time.monotonic() - begin < seconds:
            start = time.monotonic()
            elapsed, last = start - last, start
            with viewer.lock():
                demo_targets(controller, data.time)
                steps = clock.steps(elapsed)
                if steps:
                    controller.command(data, dt=steps * model.opt.timestep)
                    advance_physics(model, data, controller, steps=steps)
                    recorder.sample(data, samples={}, fresh={side: False for side in SIDES})
                recorder.maybe_render(data)
            if headset:
                lines = headset_lines({side: True for side in SIDES}, mapper, controller,
                                      recorder=recorder, data=data)
                with viewer.lock():
                    headset.maybe_send(data, viewer.cam, lines, reuse=recorder.latest_frame("operator"))
            viewer.sync()
            time.sleep(max(0, .01 - (time.monotonic() - start)))
        with contextlib.redirect_stdout(io.StringIO()):
            directory = recorder.stop(data, "user_stop")
        recorder.close()
    if headset:
        headset.close()
    with h5py.File(Path(directory) / "episode.h5") as file:
        period = np.diff(file["time/wall"][...]) * 1000
        video = json.loads(file.attrs["metadata"])["video"]
    return period, video, headset.stats() if headset else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=(*CAMERAS, "none"), default="follow")
    parser.add_argument("--size", type=video_size, default=(960, 540))
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--budget-ms", type=float, default=16)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        period, video, stats = run(args.camera, args.size, args.fps, args.seconds, folder, args.budget_ms / 1000)
    print(f"headset {args.camera} {args.size[0]}x{args.size[1]} @ {args.fps:g} fps, budget {args.budget_ms:g} ms, "
          "recording 30 fps video")
    print(f"loop period ms: median {np.median(period):.1f}  p95 {np.percentile(period, 95):.1f}  "
          f"max {period.max():.1f}")
    print(f"recorder render ms: mean {video['render_ms']['mean']:.1f}, dropped {video['dropped_frames']}")
    if stats:
        render = stats["render_ms"] or {}
        print(f"headset frames sent {stats['sent']} ({stats['sent'] / args.seconds:.1f}/s), "
              f"reused {stats['reused']}, over budget {stats['skipped']}, "
              f"render+overlay ms median {render.get('median', float('nan')):.1f}")


if __name__ == "__main__":
    main()
