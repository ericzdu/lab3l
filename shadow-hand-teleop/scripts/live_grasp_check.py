"""Headless live-grasp check: close the right hand only through finger_targets().

Drives the right wrist to the scripted demo's wrap pose (same targets and
limits as scripted_cup_to_tray), then closes the fingers with finger_targets()
fed a synthetic human grasp (open -> fist, thumb across), slew-limited exactly
like view_dual_arms' live loop. It never uses fist_targets. Then it lifts 5 cm
and holds 2 s, and reports how far the cup rose, slipped in the hand and tilted.

    .venv/bin/python scripts/live_grasp_check.py
    .venv/bin/python scripts/live_grasp_check.py --grasp-assist
"""
from __future__ import annotations

import argparse

import numpy as np

from hand_retargeting import apply_targets, finger_targets, synthetic_grasp_skeleton
from scripted_cup_to_tray import SIDE, SLOW, Script, palm_inward, waypoints

CLOSE_SECONDS = .9
LIFT = .05
HOLD_SECONDS = 2.
LIVE_SLEW = 7.5   # rad/s, as view_dual_arms applies live finger targets


class LiveHandScript(Script):
    """Script whose right fingers follow finger_targets(skeleton) each 10 ms tick."""

    def __init__(self, *args, assist=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skeleton = synthetic_grasp_skeleton(0.)
        self.assist = assist
        self.assist_seen = False

    def step(self):
        fingers = {name: value for name, value in finger_targets(self.skeleton).items()
                   if name not in ("WRJ1", "WRJ2")}
        desired = self.data.ctrl.copy()
        apply_targets(desired, fingers, self.hands[SIDE], self.model.actuator_ctrlrange)
        ids = list(self.hands[SIDE].values())
        self.data.ctrl[ids] += np.clip(desired[ids] - self.data.ctrl[ids], -LIVE_SLEW * .01, LIVE_SLEW * .01)
        super().step()
        if self.assist is not None:
            for kind, payload in self.assist.update(self.data):
                if self.recorder is not None:
                    self.recorder.event(self.data, kind, **payload)
            self.assist_seen |= bool(self.assist.active[SIDE])


def cup_tilt(data):
    """Degrees between the cup's axis and world up."""
    axis = data.body("cup").xmat.reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(axis[2], -1, 1))))


def in_wrist(data, controller):
    site = controller.sites[SIDE]
    rotation = data.site_xmat[site].reshape(3, 3)
    return rotation.T @ (data.body("cup").xpos - data.site_xpos[site])


def measure(grasp_assist=False, offset=(0., 0., 0.), record_dir=None):
    """Run once; return dict(lifted, slip, tilt) in metres/degrees, plus assist use.

    offset (m) shifts the wrap pose, as an imprecise live approach would.
    record_dir: also record (and auto-label) the run as a scripted_demo
    episode there; its folder is returned as "directory".
    """
    from view_dual_arms import setup
    model, data, controller, hands = setup("human", workspace=True)
    assist = None
    if grasp_assist:
        from grasp_assist import GraspAssist
        assist = GraspAssist(model, hands)
    recorder = None
    if record_dir is not None:
        from episode_recorder import EpisodeRecorder
        recorder = EpisodeRecorder(model, controller, hands, record_dir=record_dir, source="scripted_demo",
                                   video=False, assist=assist)
        recorder.start(data)
    script = LiveHandScript(model, data, controller, hands, recorder=recorder, assist=assist)
    points = waypoints(controller)
    for name in ("pre_grasp_high", "pre_grasp", "grasp"):
        points[name] = points[name] + offset
    script.hold(.5)
    rest = data.body("cup").xpos.copy()
    script.move(points["clear"])
    script.rotation = palm_inward(controller)
    script.hold(1.2)
    script.move(points["pre_grasp_high"])
    script.move(points["pre_grasp"])
    script.move(points["grasp"], speed=SLOW)
    begin = data.time
    while data.time - begin < CLOSE_SECONDS:
        script.skeleton = synthetic_grasp_skeleton(min(1., (data.time - begin) / CLOSE_SECONDS))
        script.step()
    script.skeleton = synthetic_grasp_skeleton(1.)
    script.hold(.4)
    held = in_wrist(data, controller)
    script.move(points["grasp"] + [0, 0, LIFT], speed=SLOW, settle=0.)
    script.hold(HOLD_SECONDS)
    directory = None
    if recorder is not None:
        if assist is not None:
            for kind, payload in assist.release_all(data, "user_stop"):
                recorder.event(data, kind, **payload)
        directory = recorder.stop(data, "user_stop")
        recorder.close()
    return {"directory": directory,
            "lifted": float(data.body("cup").xpos[2] - rest[2]),
            "slip": float(np.linalg.norm(in_wrist(data, controller) - held)),
            "tilt": cup_tilt(data),
            "thj4": float(data.ctrl[hands[SIDE]["THJ4"]]),
            "assist_used": script.assist_seen}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grasp-assist", action="store_true")
    args = parser.parse_args(argv)
    result = measure(args.grasp_assist)
    print(f"lifted {result['lifted'] * 100:.1f} cm, slip {result['slip'] * 100:.2f} cm, "
          f"tilt {result['tilt']:.1f} deg, THJ4 command {result['thj4']:.2f}"
          + (f", assist {'used' if result['assist_used'] else 'not used'}" if args.grasp_assist else ""))


if __name__ == "__main__":
    main()
