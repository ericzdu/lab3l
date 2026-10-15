"""Replay recorded Shadow Hand states from an NPZ or HDF5 demonstration."""
import argparse
import math
import glfw
import numpy as np
import time

import mujoco
import mujoco.viewer
import mujoco_menagerie as menagerie

from collect_mock_demo import PROJECT_ROOT  # Also configures the model cache.
from replay_data import load_demo, validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=str, help=".h5, .hdf5, or .npz demonstration")
    parser.add_argument("--speed", type=float, default=1, help="Playback multiplier (default: 1)")
    parser.add_argument("--headless", action="store_true", help="Validate and restore all frames without a viewer")
    parser.add_argument("--demo", default="demo_0", help="Holos demonstration group (default: demo_0)")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    comparison = None
    try:
        arrays = load_demo(args.file, args.demo)
        if "qpos" not in arrays:
            from view_dual_shadow_hands import build_model
            from holos_replay import retarget_holos
            source = arrays
            arrays = retarget_holos(source, build_model())
            model = build_model(moving=True)
            from replay_comparison import ReplayComparison
            comparison = ReplayComparison(source, model)
        else:
            model = menagerie.load("shadow_hand")
        validate_model(arrays, model)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    data = mujoco.MjData(model)
    if comparison:
        data.mocap_pos[:] = [0, 0, -10]

    def restore(index):
        data.qpos[:] = arrays["qpos"][index]
        data.qvel[:] = arrays["qvel"][index]
        data.ctrl[:] = arrays["action"][index]
        data.time = float(arrays["timestamp"][index])
        if comparison:
            comparison.update(data, index)
        mujoco.mj_forward(model, data)
        if comparison:
            comparison.advance_smoke(data, data.time)

    if args.headless:
        for index in range(len(arrays["timestamp"])):
            restore(index)
    else:
        restore(0)
        def key_callback(key):
            if comparison and key == glfw.KEY_S:
                comparison.smoke = not comparison.smoke
                print("Rising smoke: " + ("ON" if comparison.smoke else "OFF"))

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            if comparison:
                with viewer.lock():
                    viewer.cam.lookat[:] = comparison.lookat
                    viewer.cam.distance = comparison.distance
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -15
                    comparison.draw(viewer.user_scn, 0)
                print("Left: recorded skeletons. Right: moving Shadow Hands. S toggles smoke. Close window to stop.")
            started = time.monotonic()
            first = arrays["timestamp"][0]
            last_index = -1
            timestamps = arrays['timestamp']
            # Render at 30 Hz rather than submitting every capture packet.
            # When a frame is slow, select the current recorded time instead
            # of building up a queue of overdue frames.
            while viewer.is_running():
                frame_start = time.monotonic()
                recorded_time = first + (frame_start - started) * args.speed
                index = min(len(timestamps) - 1,
                            max(0, int(np.searchsorted(timestamps, recorded_time, side='right') - 1)))
                if index != last_index:
                    with viewer.lock():
                        restore(index)
                        if comparison:
                            comparison.draw(viewer.user_scn, index)
                    last_index = index
                viewer.sync()
                if index == len(timestamps) - 1:
                    break
                time.sleep(max(0., 1 / 30 - (time.monotonic() - frame_start)))
    if comparison and hasattr(comparison, "plume"):
        print(f"Smoke surface interactions: {comparison.plume.collisions}")
    print(f"Replay finished: {args.file}")


if __name__ == "__main__":
    main()
