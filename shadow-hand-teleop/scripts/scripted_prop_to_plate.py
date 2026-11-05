"""Deterministic left-hand pick and place of a prop onto the plate (apple_to_plate, block_to_plate).

Same method as scripted_cup_to_tray.py: only controller targets within the
operator's clip box and reach cap, plus FingerTransition open/close. The hand
stays palm down and closes over the object. The run is recorded as a
scripted_demo episode and labeled. Simulation only; not a policy.

    .venv/bin/python scripts/scripted_prop_to_plate.py --task apple_to_plate
    mjpython scripts/scripted_prop_to_plate.py --task block_to_plate --viewer   # macOS
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripted_cup_to_tray import SLOW, Script

SIDE = "left"
# Object centre minus wrist site when the palm-down left hand closes over it:
# ahead of the wrist (fingers +x), toward the thumb (-y), below the palm.
GRASP_OFFSET = np.array([.10, -.03, -.03])
LIFT = .08
RELEASE_HEIGHT = .008
PLATE_TASKS = ("apple_to_plate", "block_to_plate")


def perform(script, obj):
    from workspace_scene import PROPS, TABLE_TOP, object_half_height, object_start
    data, sites = script.data, script.controller.sites
    grasp = object_start(obj) - GRASP_OFFSET
    script.hold(.5)
    script.move(grasp + [0, 0, LIFT])                        # above the object
    script.move(grasp, speed=SLOW)                           # lower the palm onto it
    script.hand(closed=True)
    script.move(grasp + [0, 0, LIFT], speed=SLOW)            # lift
    # Aim from where the object actually sits in the hand, as an operator would.
    held = data.body(obj).xpos - data.site_xpos[sites[SIDE]]
    plate = data.body("plate").xpos[:2]
    place = np.r_[plate, TABLE_TOP + 2 * PROPS["plate"]["size"][1] + object_half_height(obj)] - held
    script.move(place + [0, 0, LIFT])                        # over the plate
    # Let go 8 mm above the plate. Palm down, the fingertips curl under a small
    # object, so opening them while pressing it onto the plate levers it off
    # the edge (and a press can flip the plate). The result is sensitive to
    # this height: 3 mm and 12 mm both failed for the block when tuning.
    script.move(place + [0, 0, RELEASE_HEIGHT], speed=SLOW, settle=.6)
    script.hand(closed=False)
    script.move(place + [0, 0, .10], speed=SLOW)             # retreat straight up
    script.hold(.8)                                          # let the labeler see it settle


def run(task="apple_to_plate", record_dir=None, video=False, viewer=False, auto_label=True):
    """Run once; return (episode directory, labels dict, Script)."""
    from episode_recorder import DEFAULT_RECORD_DIR, EpisodeRecorder
    from view_dual_arms import setup
    from workspace_scene import TASKS, task_props
    if task not in PLATE_TASKS:
        raise ValueError(f"task must be one of {PLATE_TASKS}")
    props = task_props(task)
    model, data, controller, hands = setup("human", workspace=True, props=props)
    recorder = EpisodeRecorder(model, controller, hands, record_dir=record_dir or DEFAULT_RECORD_DIR,
                               task=task, source="scripted_demo", wrist_mode="human",
                               workspace=True, props=props, scale=.6, video=video,
                               max_seconds=120, auto_label=auto_label)
    handle = directory = None
    try:
        if viewer:
            import mujoco.viewer
            from view_dual_arms import configure_camera
            handle = mujoco.viewer.launch_passive(model, data)
            configure_camera(handle.cam, workspace=True)
        script = Script(model, data, controller, hands, recorder, handle, side=SIDE)
        recorder.start(data)
        perform(script, TASKS[task]["object"])
        directory = recorder.stop(data, "user_stop")
    finally:
        if recorder.recording:
            recorder.stop(data, "error")
        recorder.close()
        if handle is not None:
            handle.close()
    labels = None
    if directory is not None and auto_label:
        import h5py
        from replay_data import read_episode_labels
        with h5py.File(Path(directory) / "episode.h5") as file:
            labels = read_episode_labels(file)
    return directory, labels, script


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=PLATE_TASKS, default="apple_to_plate")
    parser.add_argument("--record-dir", help="Episode folder (default: data/episodes in the project)")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--viewer", action="store_true", help="Watch in the MuJoCo viewer (mjpython on macOS)")
    args = parser.parse_args(argv)
    directory, labels, script = run(args.task, args.record_dir, video=args.video, viewer=args.viewer)
    if labels is None:
        print("No labels (labeling failed or nothing recorded).")
        return 1
    outcome = labels["outcome"]
    phases = []
    for segment in labels["segments"]:
        if not phases or phases[-1] != segment["phase"]:
            phases.append(segment["phase"])
    result = "SUCCESS" if outcome["success"] else f"FAIL ({outcome['failure_reason']})"
    print(f"Outcome: {result}, confidence {outcome['confidence']:.2f}; phases: {' > '.join(phases)}")
    print(f"Largest requested-target clipping: {script.max_reach_error * 1000:.1f} mm")
    return 0 if outcome["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
