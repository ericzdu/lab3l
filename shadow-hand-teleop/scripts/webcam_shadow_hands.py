"""Live two-hand finger-flexion control from a laptop webcam (MediaPipe).

Approximate monocular tracking: depth is inferred, so fingers pointing at the
camera bend less reliably. Same retargeter and limits as the Vision Pro demo.
"""

from __future__ import annotations

import argparse
import multiprocessing
import queue
import threading
import time
import urllib.request

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from hand_retargeting import actuator_indices, apply_targets, finger_targets
from view_dual_shadow_hands import PROJECT_ROOT, build_model


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = PROJECT_ROOT / ".cache" / "mediapipe" / "hand_landmarker.task"
WINDOW = "Webcam hand tracking (Q to quit)"
COLORS = {"left": (230, 140, 40), "right": (40, 140, 255)}  # BGR: blue, orange


def mediapipe_skeleton(points: np.ndarray) -> np.ndarray:
    """Place MediaPipe's 21 landmarks into the 25-joint layout finger_targets reads.

    MediaPipe has no finger metacarpal joints, so the wrist stands in for them:
    the first finger bone runs wrist -> knuckle. Only positions are filled.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3):
        raise ValueError("Expected 21 MediaPipe landmarks")
    joints = np.repeat(np.eye(4)[None], 25, axis=0)
    joints[0:5, :3, 3] = points[0:5]  # wrist, thumb CMC, MCP, IP, tip
    for finger in range(4):
        source = 5 + 4 * finger
        start = 5 + 5 * finger
        joints[start, :3, 3] = points[0]
        joints[start + 1 : start + 5, :3, 3] = points[source : source + 4]
    return joints


def track_hands(camera, model_path, output, stop):
    """Child process: owns the camera and the OpenCV preview on its main thread."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    capture = cv2.VideoCapture(camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {camera} (check macOS camera permission)")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            started = time.monotonic()
            timestamp = -1
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped delivering frames")
                # MediaPipe labels handedness assuming a mirrored selfie image.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(timestamp + 1, int((time.monotonic() - started) * 1000))
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                hands = {}
                height, width = frame.shape[:2]
                for handedness, world, image_points in zip(
                        result.handedness, result.hand_world_landmarks, result.hand_landmarks):
                    side = handedness[0].category_name.lower()
                    hands[side] = np.array([[p.x, p.y, p.z] for p in world])
                    pixels = [(int(p.x * width), int(p.y * height)) for p in image_points]
                    for connection in connections:
                        cv2.line(frame, pixels[connection.start], pixels[connection.end], COLORS[side], 2)
                    for pixel in pixels:
                        cv2.circle(frame, pixel, 3, COLORS[side], -1)
                    cv2.putText(frame, side, pixels[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[side], 2)
                try:
                    output.put_nowait(hands)
                except queue.Full:
                    pass  # The controller is behind; drop this frame rather than lag.
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except Exception as exc:
        output.put(str(exc))
    finally:
        capture.release()
        cv2.destroyAllWindows()


class WebcamBridge:
    def __init__(self, camera, smoothing):
        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue(maxsize=2)
        self.process_stop = context.Event()
        self.process = context.Process(target=track_hands, daemon=True,
                                       args=(camera, MODEL_PATH, self.queue, self.process_stop))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.samples = {}
        self.error = None
        self.smoothing = smoothing
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.process.start()
        self.thread.start()

    def run(self):
        filtered = {}
        while not self.stop.is_set():
            try:
                hands = self.queue.get(timeout=0.1)
            except queue.Empty:
                if not self.process.is_alive():
                    with self.lock:
                        self.error = "Camera preview closed"
                    return
                continue
            if isinstance(hands, str):
                with self.lock:
                    self.error = hands
                return
            packet = {}
            for side, points in hands.items():
                try:
                    targets = finger_targets(mediapipe_skeleton(points))
                except ValueError:
                    continue
                previous = filtered.get(side)
                if previous is not None:
                    targets = {name: self.smoothing * previous[name] + (1 - self.smoothing) * value
                               for name, value in targets.items()}
                packet[side] = (time.monotonic(), targets)
            # A hand that drops out restarts its filter when it reappears.
            filtered = {side: entry[1] for side, entry in packet.items()}
            with self.lock:
                self.samples = packet

    def snapshot(self):
        with self.lock:
            return dict(self.samples), self.error

    def close(self):
        self.stop.set()
        self.process_stop.set()
        self.thread.join(timeout=1)
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading MediaPipe hand model to {MODEL_PATH} ...", flush=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.rename(MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="Target smoothing 0 (none) to <1 (heavy); default 0.5")
    parser.add_argument("--seconds", type=float, default=0, help="Optional time limit; 0 runs until closed")
    args = parser.parse_args()
    if not 0 <= args.smoothing < 1:
        parser.error("--smoothing must be in [0, 1)")
    ensure_model()

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    indices = {side: actuator_indices(model, side) for side in ("left", "right")}
    bridge = WebcamBridge(args.camera, args.smoothing)
    state = {"paused": False}

    def key_callback(key):
        if key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            print("PAUSED" if state["paused"] else "RESUMED", flush=True)

    print("LIVE webcam finger-flexion demo: left hand -> left robot; right -> right.", flush=True)
    print("Fixed wrists; approximate monocular flexion only, NOT full pose.", flush=True)
    print("No recording. Camera frames stay on this Mac and are not saved.", flush=True)
    print("SPACE (MuJoCo window) pauses. Close either window or press Q in the preview to stop.", flush=True)
    bridge.start()
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
                if error:
                    print("Stopping: " + error, flush=True)
                    break
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
                if status != previous_status:
                    print(status, flush=True)
                    previous_status = status
                with viewer.lock():
                    # Neither hand visible or paused: freeze the simulated state.
                    # A single missing hand holds its last actuator target.
                    if any(fresh.values()) and not state["paused"]:
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
        print(f"Stopped. Fresh mapped frames: {counts}. No data saved.", flush=True)


if __name__ == "__main__":
    main()
