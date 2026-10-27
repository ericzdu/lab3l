"""Live-style grasp: fingers closed only through finger_targets(), no headset.

The right wrist goes to the scripted demo's wrap pose; a synthetic human grasp
(open -> fist, thumb across) is retargeted and slew-limited as in the live
loop. Lift 5 cm, hold 2 s: the cup must stay in hand with and without
--grasp-assist. See live_grasp_check.py.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import mujoco
import numpy as np

from arm_control import SIDES
from live_grasp_check import measure

MIN_LIFT, MAX_SLIP, MAX_TILT = .04, .01, 20.


class LiveGraspTests(unittest.TestCase):
    def assert_held(self, result):
        self.assertGreaterEqual(result["lifted"], MIN_LIFT, result)
        self.assertLess(result["slip"], MAX_SLIP, result)
        self.assertLess(result["tilt"], MAX_TILT, result)

    def test_retargeted_grasp_holds_cup(self):
        result = measure()
        self.assert_held(result)
        self.assertGreaterEqual(result["thj4"], 1.)    # thumb opposed from tracking
        self.assertFalse(result["assist_used"])

    def test_assisted_grasp_holds_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = measure(grasp_assist=True, record_dir=folder)
            self.assert_held(result)
            self.assertTrue(result["assist_used"])
            directory = Path(result["directory"])
            with h5py.File(directory / "episode.h5") as file:
                active = file["assist/active"][...]
                metadata = json.loads(file.attrs["metadata"])
                events = json.loads(file.attrs["events"])
                arrays = {name: file[name][...] for name in ("time/sim", "contacts/hand_cup",
                          "contacts/arm_cup", "contacts/cup_table", "contacts/cup_tray",
                          "scene/cup_pose", "scene/cup_linvel", "scene/cup_angvel",
                          "contacts/hand_hand", "robot/wrist_pos", "input/fresh")}
            self.assertEqual(active.shape[1], 2)
            self.assertTrue(active[:, SIDES.index("right")].any())
            self.assertFalse(active[:, SIDES.index("left")].any())
            self.assertTrue(metadata["grasp_assist"])
            self.assertEqual(metadata["grasp_assist_activations"], 1)
            self.assertEqual([e["kind"] for e in events if e["kind"].startswith("assist")],
                             ["assist_on", "assist_off"])
            # The labeler sees only simulated state: same labels without the assist flags.
            from episode_labels import label_episode, labels_json
            plain = dict(metadata, grasp_assist=False, grasp_assist_activations=0, labels=None)
            labels, unassisted = (dict(labels_json(label_episode(arrays, m)), created_utc=None)
                                  for m in (metadata, plain))
            self.assertEqual(labels, unassisted)
            from summarize_episodes import episode_row
            self.assertEqual(episode_row(metadata)["assist"], "on (1)")
            self.assertEqual(episode_row({})["assist"], "")


class GraspAssistTests(unittest.TestCase):
    def setUp(self):
        from grasp_assist import GraspAssist
        from view_dual_arms import setup
        self.model, self.data, self.controller, self.hands = setup("human", workspace=True)
        self.assist = GraspAssist(self.model, self.hands)

    def test_welds_exist_inactive_without_changing_sizes(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (69, 64, 50))
        self.assertEqual(self.model.neq, 2)
        self.assertFalse(self.data.eq_active.any())
        self.assertFalse(self.model.eq_active0.any())

    def touch(self, thumb=True, finger=True, anything=True):
        touching = {side: (False, False, False) for side in SIDES}
        touching["right"] = (thumb, finger, anything)
        self.assist.touching = lambda data: touching

    def close(self, amount):
        for prefix in ("FF", "MF", "RF", "LF"):
            self.data.ctrl[self.hands["right"][prefix + "J3"]] = amount * np.pi / 2
            self.data.ctrl[self.hands["right"][prefix + "J0"]] = amount * np.pi

    def tick(self, seconds=.01):
        events = []
        for _ in range(int(round(seconds / .01))):
            self.data.time += .01
            events += self.assist.update(self.data)
        return events

    def test_engages_only_with_thumb_finger_and_settled_closure(self):
        self.close(.8)
        for touch in ((False, True, True), (True, False, True)):
            self.touch(*touch)
            self.assertEqual(self.tick(.5), [])
        self.touch()
        self.close(.3)
        self.assertEqual(self.tick(.5), [])
        self.close(.8)
        self.assertEqual(self.tick(.05), [])            # still settling after the jump
        events = self.tick(.3)
        self.assertEqual([kind for kind, _ in events], ["assist_on"])
        self.assertTrue(self.data.eq_active[self.assist.welds["right"]])

    def test_engaging_does_not_move_the_cup(self):
        mujoco.mj_forward(self.model, self.data)
        before = self.data.body("cup").xpos.copy()
        self.assist._engage(self.data, "right", .8)
        mujoco.mj_forward(self.model, self.data)
        weld = [i for i in range(self.data.nefc)
                if self.data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        self.assertLess(np.abs(self.data.efc_pos[weld]).max(), 1e-6)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.assertLess(np.linalg.norm(self.data.body("cup").xpos - before), 1e-3)

    def engaged(self):
        self.touch()
        self.close(.8)
        self.tick(.3)
        self.assertTrue(self.assist.active["right"])

    def test_hysteresis_release(self):
        self.engaged()
        self.close(.4)                                  # between OFF and ON: keep holding
        self.assertEqual(self.tick(.5), [])
        self.close(.3)
        self.assertEqual(self.tick(), [("assist_off", {"side": "right", "reason": "opened"})])
        self.assertFalse(self.data.eq_active.any())

    def test_release_after_contact_lost(self):
        self.engaged()
        self.touch(False, False, False)
        self.assertEqual(self.tick(.25), [])
        self.assertEqual(self.tick(.1), [("assist_off", {"side": "right", "reason": "contact_lost"})])

    def test_reset_releases(self):
        from arm_control import WristMapper
        from view_dual_arms import reset_scene
        self.engaged()
        reset_scene(self.data, self.controller, WristMapper(), assist=self.assist)
        self.assertFalse(self.assist.active["right"])
        self.assertFalse(self.data.eq_active.any())

    def test_one_hand_at_a_time(self):
        self.engaged()
        touching = {side: (True, True, True) for side in SIDES}
        self.assist.touching = lambda data: touching
        for prefix in ("FF", "MF", "RF", "LF"):
            self.data.ctrl[self.hands["left"][prefix + "J3"]] = .8 * np.pi / 2
            self.data.ctrl[self.hands["left"][prefix + "J0"]] = .8 * np.pi
        self.assertEqual(self.tick(.5), [])
        self.assertFalse(self.assist.active["left"])

    def test_cli_needs_workspace(self):
        import sys
        from unittest import mock
        from view_dual_arms import parse_args
        with mock.patch.object(sys, "argv", ["view_dual_arms.py", "--grasp-assist"]), \
                self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args()
        with mock.patch.object(sys, "argv", ["view_dual_arms.py", "--wrist", "human", "--workspace"]):
            self.assertFalse(parse_args().grasp_assist)


if __name__ == "__main__":
    unittest.main()
