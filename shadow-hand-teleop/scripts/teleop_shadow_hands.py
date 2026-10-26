"""Live Vision Pro finger flexion -> two fixed Shadow Hands in a Mac viewer.

No recording, no camera/microphone configuration, and no trained policy.
Run with mjpython on macOS. Streaming runs separately so waiting cannot freeze UI.
"""

from __future__ import annotations

import argparse
import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from hand_retargeting import actuator_indices, apply_targets, finger_targets
from view_dual_shadow_hands import build_model


class TrackingBridge:
    """Latest per-side packet, read by the viewer loop.

    Each side maps to (receipt time, finger targets). With with_wrist=True
    the tuple continues: [2] validated wrist transform, [3] forearm vector or
    None, [4] raw (25,4,4) finger skeleton copy, [5] raw 27-point arm skeleton
    copy or None, [6] bridge-wide packet sequence number. Indices 0-3 are
    relied on elsewhere and never change.
    """
    def __init__(self, factory, address, with_wrist=False):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.streamer = None
        self.samples = {}
        self.error = None
        self.factory = factory
        self.address = address
        self.with_wrist = with_wrist
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            self.streamer = self.factory(ip=self.address, record=False,
                                         benchmark_quiet=True, verbose=False)
            previous = None
            sequence = -1
            while not self.stop.is_set():
                sample = self.streamer.get_latest()
                if sample is not None:
                    # avp_stream 2.51 replaces this dictionary for each packet.
                    # Re-reading an old cached sample must not reset freshness.
                    raw = sample._raw
                    if raw is not previous:
                        previous = raw
                        sequence += 1
                        packet = {}
                        for side in ("left", "right"):
                            try:
                                entry = (time.monotonic(), finger_targets(sample[f"{side}_fingers"]))
                                if self.with_wrist:
                                    from arm_control import wrist_position
                                    wrist = np.asarray(sample[f"{side}_wrist"], dtype=float).copy()
                                    wrist_position(wrist)
                                    entry = (*entry, wrist)
                                    from forearm_tracking import forearm_vector
                                    # Optional: absence of 27-point data must
                                    # not disable valid wrist/finger streaming.
                                    try:
                                        arm = sample[f"{side}_arm"]
                                    except KeyError:
                                        arm = None
                                    entry = (*entry, forearm_vector(arm, wrist))
                                    # Raw input for recording; retargeters change,
                                    # so recordings must keep what was received.
                                    fingers = np.array(sample[f"{side}_fingers"], dtype=float)
                                    raw_arm = None
                                    if arm is not None:
                                        try:
                                            raw_arm = np.array(arm, dtype=float)
                                        except (ValueError, TypeError):
                                            raw_arm = None
                                    entry = (*entry, fingers, raw_arm, sequence)
                                packet[side] = entry
                            except (ValueError, KeyError, TypeError):
                                pass
                        with self.lock:
                            self.samples = packet
                self.stop.wait(0.008)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def snapshot(self):
        with self.lock:
            return dict(self.samples), self.error

    def close(self):
        self.stop.set()
        if self.streamer is not None:
            self.streamer.cleanup()
        self.thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="Tracking Streamer IP or CURRENT remote room code")
    parser.add_argument("--seconds", type=float, default=0, help="Optional time limit; 0 runs until closed")
    args = parser.parse_args()
    # Import on the main thread: this library installs signal handlers.
    from avp_stream import VisionProStreamer

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    indices = {side: actuator_indices(model, side) for side in ("left", "right")}
    bridge = TrackingBridge(VisionProStreamer, args.address)
    state = {"paused": False}

    def key_callback(key):
        if key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            print("PAUSED" if state["paused"] else "RESUMED", flush=True)

    print("LIVE finger-flexion demo: left hand -> left robot; right -> right.", flush=True)
    print("Fixed wrists; approximate flexion only, NOT full pose or pinch calibration.", flush=True)
    print("No recording. No camera/microphone. Viewer is on the Mac, not in the headset.", flush=True)
    print("SPACE pauses. Close the MuJoCo window to disconnect.", flush=True)
    bridge.thread.start()
    started = time.monotonic()
    previous_status = None
    counts = {"left": 0, "right": 0}
    last_received = {"left": None, "right": None}
    target = data.ctrl.copy()
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.18, 0, 0.03]
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -12
            while viewer.is_running():
                frame_start = time.monotonic()
                if args.seconds and frame_start - started >= args.seconds:
                    break
                samples, error = bridge.snapshot()
                fresh = {}
                for side in indices:
                    entry = samples.get(side)
                    fresh[side] = entry is not None and frame_start - entry[0] < 0.5
                    if fresh[side] and not state["paused"]:
                        apply_targets(target, entry[1], indices[side], model.actuator_ctrlrange)
                        if entry[0] != last_received[side]:
                            counts[side] += 1
                            last_received[side] = entry[0]
                status = " | ".join(f"{side}: {'LIVE' if fresh[side] else 'WAIT / HOLD'}" for side in indices)
                if state["paused"]:
                    status = "PAUSED"
                if error:
                    status = "Connection error: " + error
                if status != previous_status:
                    print(status, flush=True)
                    previous_status = status
                with viewer.lock():
                    # No fresh packets or paused: freeze the simulated state.
                    # Per-hand commands hold their last target if only one is stale.
                    if any(fresh.values()) and not state["paused"] and not error:
                        for _ in range(5):
                            delta = np.clip(target - data.ctrl, -0.015, 0.015)
                            data.ctrl[:] += delta
                            mujoco.mj_step(model, data)
                    if not np.isfinite(data.qpos).all():
                        raise RuntimeError("Non-finite simulation state; stopping")
                viewer.sync()
                time.sleep(max(0, 0.01 - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        print(f"Stopped. Fresh mapped packets: {counts}. No data saved.", flush=True)


if __name__ == "__main__":
    main()
