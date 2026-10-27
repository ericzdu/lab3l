"""Replay recorded Shadow Hand states from an NPZ or HDF5 demonstration.

Also replays lab3l.episode recordings of the human-arm workspace (arms, hands
and cup) and, if the episode is labeled, prints phase transitions.
"""
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


def episode_model(metadata):
    """Build the recorded scene and require identical joint/actuator layout."""
    scene = metadata.get("scene", {})
    if scene.get("wrist_mode") != "human":
        raise ValueError(f"Unsupported episode wrist mode: {scene.get('wrist_mode')!r}")
    from human_arms import build_human_model
    model = build_human_model(workspace=bool(scene.get("workspace")))
    recorded = metadata.get("model", {})
    for kind, count, name in (("joint_names", model.njnt, model.joint),
                              ("actuator_names", model.nu, model.actuator)):
        current = [name(i).name for i in range(count)]
        if recorded.get(kind) != current:
            raise ValueError(f"Episode {kind} do not match the current human-arm model; "
                             "it was recorded with a different scene version")
    warning = prop_layout_warning(metadata)
    if warning:
        print(warning, flush=True)
    return model


def prop_layout_warning(metadata):
    """Warn when the episode's table/cup/tray differ from the current scene.

    Replay rebuilds the model from current code: the robot replay stays valid
    (same nq/nv/nu), but the table and tray are drawn at their new positions.
    """
    import workspace_scene
    from episode_labels import scene_geometry
    recorded = scene_geometry(metadata)
    if not recorded["table_top"] or not metadata.get("scene", {}).get("workspace", True):
        return None
    current = {"table_top": workspace_scene.TABLE_TOP, "cup_start": workspace_scene.CUP_START,
               "tray_center": workspace_scene.TRAY_CENTER}
    changed = [key for key, value in current.items()
               if not np.allclose(recorded[key], value, atol=1e-6)]
    if not changed:
        return None
    return ("WARNING: this episode was recorded with scene version "
            f"{recorded['scene_version']} ({', '.join(changed)} differ from the current scene). "
            "The robot and cup replay as recorded, but the table and tray are shown at their "
            "NEW positions, so the cup may appear to float or sink.")


class PhaseAnnouncer:
    """Print labeled phase changes as playback crosses them."""
    def __init__(self, labels):
        self.segments = (labels or {}).get("segments") or []
        self.outcome = (labels or {}).get("outcome")
        self.next = 0
        self.current = ""

    def update(self, time):
        while self.next < len(self.segments) and self.segments[self.next]["start_t"] <= time + 1e-9:
            segment = self.segments[self.next]
            self.current = f"{segment['phase']} ({segment.get('side') or '-'}, conf {segment['confidence']:.2f})"
            print(f"t={segment['start_t']:6.2f}s  phase: {self.current}", flush=True)
            self.next += 1
        return self.current

    def finish(self):
        if not self.outcome:
            return
        if self.outcome.get("success") is None:
            print(f"Outcome: {self.outcome.get('outcome', 'not evaluated')}")
        else:
            result = "SUCCESS" if self.outcome["success"] else f"FAIL ({self.outcome['failure_reason']})"
            print(f"Outcome: {result}, confidence {self.outcome['confidence']:.2f}")


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
        if "_metadata" in arrays:
            model = episode_model(arrays["_metadata"])
        elif "qpos" not in arrays:
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
    phases = PhaseAnnouncer(arrays.get("_labels"))
    if "_metadata" in arrays and not arrays.get("_labels"):
        print("Episode is not labeled; run scripts/label_episode.py to add phases.")
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
            phases.update(data.time)
    else:
        restore(0)
        def key_callback(key):
            if comparison and key == glfw.KEY_S:
                comparison.smoke = not comparison.smoke
                print("Rising smoke: " + ("ON" if comparison.smoke else "OFF"))

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            if "_metadata" in arrays:
                from view_dual_arms import configure_camera
                with viewer.lock():
                    configure_camera(viewer.cam, workspace=True)
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
                    phase = phases.update(data.time)
                    if phase and hasattr(viewer, "set_texts"):
                        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Phase", phase))
                    last_index = index
                viewer.sync()
                if index == len(timestamps) - 1:
                    break
                time.sleep(max(0., 1 / 30 - (time.monotonic() - frame_start)))
    if comparison and hasattr(comparison, "plume"):
        print(f"Smoke surface interactions: {comparison.plume.collisions}")
    phases.finish()
    print(f"Replay finished: {args.file}")


if __name__ == "__main__":
    main()
