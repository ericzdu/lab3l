"""Deterministic left-hand demos of the prop tasks: onto the plate, stacking, setting the table.

Tasks: apple_to_plate, block_to_plate, block_on_big_block, set_the_table
(apple then block onto the plate, one after the other), and the same four
with _real: the real 3D models (workspace_scene.PROPS real_*).

Same method as scripted_cup_to_tray.py: only controller targets within the
operator's clip box and reach cap, plus FingerTransition open/close. The hand
stays palm down and closes over the object. The run is recorded as a
scripted_demo episode and labeled. Simulation only; not a policy.

    .venv/bin/python scripts/scripted_prop_tasks.py --task block_on_big_block
    mjpython scripts/scripted_prop_tasks.py --task set_the_table --viewer   # macOS
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
# Per-object exceptions for the scripted hand (not used by the scene or labels):
# "offset" replaces GRASP_OFFSET; "at" moves the grasp point from the object's
# bounding-box centre, in its frame. The SAM 3D apple is smaller than the
# props GRASP_OFFSET was tuned on, so the palm closes 1 cm lower; a banana's
# box centre lies inside its curve, so the hand closes on the fruit 30 mm nearer.
SCRIPTED_GRASP = {"sam_apple": {"offset": (.10, -.03, -.02)},
                  "banana": {"at": (-.030, 0)}}
# Tasks whose scripted demo uses grasp assist by default (flagged in the
# episode). This palm-down fist cannot hold the team's fruit reliably on its
# own: the 40 mm banana is too low (the fingers reach the table before they
# curl under it) and the 61 mm apple slips out in most nearby poses. With the
# assist both succeed across every pose tried.
ASSISTED_TASKS = ("sam_apple_to_plate", "banana_to_plate")
RELEASE_HEIGHT = .006
SCRIPTED_TASKS = ("apple_to_plate", "block_to_plate", "block_on_big_block", "set_the_table",
                  "apple_to_plate_real", "block_to_plate_real", "block_on_big_block_real",
                  "set_the_table_real", "sam_apple_to_plate", "banana_to_plate")
# Where each object goes on its target, relative to the target's centre. The
# plate is too small for the apple and block side by side at the centre:
# apple on the near half, block on the far half (tolerates about +/-1 cm).
PLACE_OFFSETS = {"set_the_table": {"apple": (0, -.03), "block": (0, .035)},
                 "set_the_table_real": {"real_apple": (0, -.03), "real_block": (0, .035)}}


def place_object(script, obj, target, offset=(0, 0)):
    """Pick `obj` up palm down and set it on `target` (plate or big block)."""
    from workspace_scene import object_half_height, object_start
    data, sites = script.data, script.controller.sites
    # Lift high enough that the object's bottom clears the target's top.
    lift = max(LIFT, 2 * object_half_height(target) + .02)
    special = SCRIPTED_GRASP.get(obj, {})
    at = np.r_[special.get("at", (0, 0)), 0]
    grasp_offset = np.asarray(special.get("offset", GRASP_OFFSET))
    grasp = np.r_[data.body(obj).xpos[:2], object_start(obj)[2]] + at - grasp_offset
    script.move(grasp + [0, 0, LIFT])                        # above the object
    script.move(grasp, speed=SLOW)                           # lower the palm onto it
    script.hand(closed=True)
    script.move(grasp + [0, 0, lift], speed=SLOW)            # lift
    # Aim from where the object actually sits in the hand, as an operator would.
    held = data.body(obj).xpos - data.site_xpos[sites[SIDE]]
    top = data.body(target).xpos[2] + object_half_height(target)
    place = np.r_[data.body(target).xpos[:2] + offset, top + object_half_height(obj)] - held
    script.move(place + [0, 0, LIFT])                        # over the target
    # Let go 6 mm above the target, the middle of the range every task passed
    # with +/-3 mm placement error: 4-8 mm. set_the_table bounds it (from 9 mm
    # the dropped block knocks the apple or itself off the small plate); the
    # single tasks work across 3-16 mm.
    script.move(place + [0, 0, RELEASE_HEIGHT], speed=SLOW, settle=.6)
    script.hand(closed=False)
    script.move(place + [0, 0, .10], speed=SLOW)             # retreat straight up


def perform(script, task):
    from workspace_scene import TASKS, task_steps
    offsets = PLACE_OFFSETS.get(task, {})
    script.hold(.5)
    for step in task_steps(task):
        obj, target = TASKS[step]["object"], TASKS[step]["target"]
        place_object(script, obj, target, offsets.get(obj, (0, 0)))
    script.hold(.8)                                          # let the labeler see it settle


def run(task="apple_to_plate", record_dir=None, video=False, viewer=False, auto_label=True,
        grasp_assist=None):
    """Run once; return (episode directory, labels dict, Script)."""
    from episode_recorder import DEFAULT_RECORD_DIR, EpisodeRecorder
    from view_dual_arms import setup
    from workspace_scene import task_props
    if task not in SCRIPTED_TASKS:
        raise ValueError(f"task must be one of {SCRIPTED_TASKS}")
    props = task_props(task)
    model, data, controller, hands = setup("human", workspace=True, props=props)
    assist = None
    if grasp_assist is None:
        grasp_assist = task in ASSISTED_TASKS
    if grasp_assist:
        from grasp_assist import GraspAssist
        from workspace_scene import task_held_objects
        assist = GraspAssist(model, hands, task_held_objects(task))
    recorder = EpisodeRecorder(model, controller, hands, record_dir=record_dir or DEFAULT_RECORD_DIR,
                               task=task, source="scripted_demo", wrist_mode="human",
                               workspace=True, props=props, scale=.6, video=video,
                               max_seconds=120, auto_label=auto_label, assist=assist)
    handle = directory = None
    try:
        if viewer:
            import mujoco.viewer
            from view_dual_arms import configure_camera
            handle = mujoco.viewer.launch_passive(model, data)
            configure_camera(handle.cam, workspace=True)
        script = Script(model, data, controller, hands, recorder, handle, side=SIDE)
        script.assist = assist
        recorder.start(data)
        perform(script, task)
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
    parser.add_argument("--task", choices=SCRIPTED_TASKS, default="apple_to_plate")
    parser.add_argument("--record-dir", help="Episode folder (default: data/episodes in the project)")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--viewer", action="store_true", help="Watch in the MuJoCo viewer (mjpython on macOS)")
    parser.add_argument("--grasp-assist", action=argparse.BooleanOptionalAction, default=None,
                        help="Weld the closed hand to the object, flagged in the episode "
                             f"(default: on only for {', '.join(ASSISTED_TASKS)})")
    args = parser.parse_args(argv)
    directory, labels, script = run(args.task, args.record_dir, video=args.video, viewer=args.viewer,
                                    grasp_assist=args.grasp_assist)
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
