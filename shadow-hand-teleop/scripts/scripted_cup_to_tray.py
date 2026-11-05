"""Deterministic right-hand cup-to-tray pick and place in the human-arm scene.

Built only from controller targets (HumanArmController.set_target and
set_rotation, within the same clip box and reach cap the operator gets) and
FingerTransition open/close commands, so it shows the scene is solvable with
the live controller limits. The run is recorded as a scripted_demo episode
and labeled with the default LabelConfig. Simulation only; not a policy.

    .venv/bin/python scripts/scripted_cup_to_tray.py            # headless
    mjpython scripts/scripted_cup_to_tray.py --viewer           # watch it (macOS)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from arm_control import rotation_matrix

SIDE = "right"
# Cup centre minus wrist site in the palm-inward grasp: the cup sits against
# the palm at the finger roots, its mid-height level with the fingers.
GRASP_OFFSET = np.array([.11, .045, -.008])
APPROACH = np.array([0, -.05, 0])   # pre-grasp: beside the cup, on the palm's side
LIFT = .08                          # m the cup is raised before transport
SPEED = .12                         # m/s along straight target paths
SLOW = .05                          # m/s for the final descent into the tray


def palm_inward(controller):
    # +90 degrees about the forearm (world +X at home), as the U key turns it.
    return rotation_matrix([np.pi / 2, 0, 0]) @ controller.home_rotations[SIDE]


class Script:
    """Drive the right arm along straight, eased target paths and step physics."""

    def __init__(self, model, data, controller, hands, recorder=None, viewer=None):
        from view_dual_arms import FingerTransition
        self.model, self.data, self.controller, self.hands = model, data, controller, hands
        self.recorder, self.viewer = recorder, viewer
        self.fingers = FingerTransition(model)
        self.rotation = controller.home_rotations[SIDE].copy()
        self.max_reach_error = 0.

    def step(self):
        from view_dual_arms import advance_physics
        self.controller.set_rotation(SIDE, self.rotation)
        self.controller.command(self.data, dt=.01)
        advance_physics(self.model, self.data, self.controller, finger_motion=self.fingers)
        if self.recorder is not None and self.recorder.recording:
            self.recorder.sample(self.data, samples={}, fresh={"left": False, "right": False})
        if self.viewer is not None:
            import time
            self.viewer.sync()
            time.sleep(.01)

    def hold(self, seconds):
        end = self.data.time + seconds
        while self.data.time < end - 1e-9:
            self.step()

    def move(self, goal, speed=SPEED, settle=.4):
        """Straight smooth path from the current target to `goal`, then settle."""
        start = self.controller.targets[SIDE].copy()
        goal = np.asarray(goal, dtype=float)
        duration = max(np.linalg.norm(goal - start) / speed, .05)
        begin = self.data.time
        while self.data.time - begin < duration:
            fraction = min(1., (self.data.time - begin) / duration)
            eased = fraction * fraction * (3 - 2 * fraction)
            self.controller.set_target(SIDE, start + eased * (goal - start))
            self.max_reach_error = max(self.max_reach_error, self.controller.reach_errors[SIDE])
            self.step()
        self.controller.set_target(SIDE, goal)
        self.max_reach_error = max(self.max_reach_error, self.controller.reach_errors[SIDE])
        self.hold(settle)

    def hand(self, closed):
        self.fingers.request(self.data, self.hands[SIDE], closed)
        self.hold((self.fingers.close_seconds if closed else self.fingers.open_seconds) + .4)


def waypoints(controller, held_offset=GRASP_OFFSET):
    """Wrist targets for each stage, from the current scene constants.

    held_offset is the cup-minus-wrist offset while carrying; the cup can
    slide a few centimetres in the hand while the fingers close.
    """
    from workspace_scene import CUP_HALF_HEIGHT, CUP_START, TABLE_TOP, TRAY_CENTER
    grasp = CUP_START - GRASP_OFFSET
    # Cup resting on the tray base (12 mm thick), centred in the tray.
    place = np.array([*TRAY_CENTER, TABLE_TOP + .012 + CUP_HALF_HEIGHT]) - held_offset
    home = controller.home[SIDE]
    return {
        # Palm down, out to the side and a little up: the turned hand's
        # little-finger edge must not land on the cup rim.
        "clear": np.array([home[0], grasp[1] + APPROACH[1] - .01, home[2] + .02]),
        "pre_grasp_high": grasp + APPROACH + [0, 0, .06],
        "pre_grasp": grasp + APPROACH,
        "grasp": grasp,
        "lift": grasp + [0, 0, LIFT],
        "over_tray": place + [0, 0, LIFT],
        # 5 mm below resting: the hand presses the cup onto the tray base
        # before the fingers open, so it does not drop or hop on release.
        "place": place + [0, 0, -.005],
        "retreat_side": place + [0, 0, -.005] + APPROACH,
        "retreat": place + APPROACH + [0, 0, .10],
    }


def perform(script):
    """The eight stages of the spec, in order."""
    points = waypoints(script.controller)
    script.hold(.5)
    script.move(points["clear"])
    script.rotation = palm_inward(script.controller)        # 1. pre-grasp, palm inward
    script.hold(1.2)
    script.move(points["pre_grasp_high"])
    script.move(points["pre_grasp"])
    script.move(points["grasp"], speed=SLOW)                 # 2. descend/slide to the wrap pose
    script.hand(closed=True)                                 # 3. close
    script.move(points["lift"], speed=SLOW)                  # 4. lift
    # Re-aim the placement from where the cup actually sits in the hand
    # (read from the simulation, as an operator would watch it).
    held = script.data.body("cup").xpos - script.data.site_xpos[script.controller.sites[SIDE]]
    points.update({k: v for k, v in waypoints(script.controller, held).items()
                   if k in ("over_tray", "place", "retreat_side", "retreat")})
    script.move(points["over_tray"])                         # 5. over the tray
    script.move(points["place"], speed=SLOW, settle=.6)      # 6. lower until it touches
    script.hand(closed=False)                                # 7. open slowly
    script.move(points["retreat_side"], speed=SLOW)          # 8. retreat
    script.move(points["retreat"])
    script.hold(.8)                                          # let the labeler see it settle


def run(record_dir=None, video=False, viewer=False, auto_label=True, quiet=False):
    """Run once; return (episode directory, labels dict, Script)."""
    from episode_recorder import DEFAULT_RECORD_DIR, EpisodeRecorder
    from view_dual_arms import setup
    model, data, controller, hands = setup("human", workspace=True)
    recorder = EpisodeRecorder(model, controller, hands, record_dir=record_dir or DEFAULT_RECORD_DIR,
                               task="cup_to_tray", source="scripted_demo", wrist_mode="human",
                               workspace=True, scale=.6, video=video, max_seconds=120,
                               auto_label=auto_label)
    handle = None
    try:
        if viewer:
            import mujoco.viewer
            from view_dual_arms import configure_camera
            handle = mujoco.viewer.launch_passive(model, data)
            configure_camera(handle.cam, workspace=True)
        script = Script(model, data, controller, hands, recorder, handle)
        recorder.start(data)
        perform(script)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-dir", help="Episode folder (default: data/episodes in the project)")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False,
                        help="Also render camera videos (default: off)")
    parser.add_argument("--viewer", action="store_true", help="Watch in the MuJoCo viewer (mjpython on macOS)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    directory, labels, script = run(args.record_dir, video=args.video, viewer=args.viewer)
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
