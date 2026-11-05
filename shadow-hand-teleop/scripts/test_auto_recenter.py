"""--auto-recenter with synthetic fresh/stale tracking (no headset, no viewer)."""
import contextlib
import io
import sys
import unittest
from unittest import mock

import numpy as np

from arm_control import TRACKING_TO_ROBOT, WristMapper
from view_dual_arms import AutoRecenter, parse_args

ANCHOR = {"left": np.array([.2, .25, .75]), "right": np.array([.2, -.25, .75])}


def wrist(position):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    return matrix


class Session:
    """Mimics the viewer: AutoRecenter.update, then invalidate stale sides."""

    def __init__(self, window=3.):
        self.mapper = WristMapper(.6)
        self.auto = AutoRecenter(window)
        self.robot = {side: anchor.copy() for side, anchor in ANCHOR.items()}
        self.recentered = []

    def calibrate(self, side, position):
        self.mapper.calibrate(side, wrist(position), self.robot[side])

    def frame(self, now, fresh, positions):
        for side, gap in self.auto.update(now, fresh, self.mapper):
            # Anchored at the robot's current wrist pose, as calibrate_side does.
            if self.mapper.recenter(side, wrist(positions[side]), self.robot[side]):
                self.auto.recentered(side, now)
                self.recentered.append((side, round(gap, 3)))
        for side in fresh:
            if not fresh[side]:
                self.mapper.invalidate(side)
            else:
                target = self.mapper.target(side, wrist(positions[side]))
                if target is not None:
                    self.robot[side] = target

    def run(self, start, end, fresh, positions, dt=.01):
        for now in np.arange(start, end, dt):
            self.frame(now, fresh, positions)


class AutoRecenterTests(unittest.TestCase):
    def test_short_gap_reanchors_without_jump_and_keeps_gains(self):
        s = Session()
        hand = {"left": np.array([.2, .3, 1.2]), "right": np.array([.2, -.3, 1.2])}
        for side in hand:
            s.calibrate(side, hand[side])
        s.mapper.position_scales["right"][1] = 1.7   # e.g. the hand-span lateral gain
        moved = dict(hand, right=hand["right"] + [.1, 0, 0])
        s.run(0, 1, {"left": True, "right": True}, moved)
        before = s.robot["right"].copy()
        # A 1.2 s Wi-Fi gap; meanwhile the operator's hand drifts 15 cm.
        drifted = dict(moved, right=moved["right"] + [0, .15, -.05])
        s.run(1, 2.2, {"left": True, "right": False}, drifted)
        self.assertNotIn("right", s.mapper.origins)
        s.frame(2.2, {"left": True, "right": True}, drifted)
        self.assertEqual([side for side, _ in s.recentered], ["right"])
        self.assertAlmostEqual(s.recentered[0][1], 1.2, delta=.02)
        # No jump: the first target after the gap is where the robot was.
        np.testing.assert_allclose(s.robot["right"], before, atol=1e-12)
        np.testing.assert_allclose(s.mapper.position_scales["right"], [.6, 1.7, .6])
        self.assertTrue(s.auto.showing("right", 2.5))
        self.assertFalse(s.auto.showing("right", 3.3))
        # Later motion uses the preserved gains, relative to the new origin.
        step = np.array([0, .1, 0])
        target = s.mapper.target("right", wrist(drifted["right"] + step))
        np.testing.assert_allclose(target - before, (TRACKING_TO_ROBOT @ step) * [.6, 1.7, .6])
        # The untouched side never recentered and keeps its default scales.
        np.testing.assert_allclose(s.mapper.position_scales["left"], [.6, .6, .6])

    def test_long_gap_stays_frozen(self):
        s = Session(window=3.)
        hand = {"left": np.array([.2, .3, 1.2]), "right": np.array([.2, -.3, 1.2])}
        s.calibrate("right", hand["right"])
        s.run(0, .5, {"left": False, "right": True}, hand)
        s.run(.5, 4., {"left": False, "right": False}, hand)
        s.run(4., 5., {"left": False, "right": True}, hand)
        self.assertEqual(s.recentered, [])
        self.assertNotIn("right", s.mapper.origins)

    def test_never_calibrated_side_is_never_auto_calibrated(self):
        s = Session()
        hand = {"left": np.array([.2, .3, 1.2]), "right": np.array([.2, -.3, 1.2])}
        s.run(0, .5, {"left": True, "right": True}, hand)
        s.run(.5, 1., {"left": False, "right": False}, hand)
        s.run(1., 1.5, {"left": True, "right": True}, hand)
        self.assertEqual(s.recentered, [])
        self.assertEqual(s.mapper.origins, {})
        # recenter() itself refuses a side with no previous calibration.
        self.assertFalse(s.mapper.recenter("left", wrist(hand["left"]), ANCHOR["left"]))
        self.assertNotIn("left", s.mapper.origins)

    def test_forget_after_pause_reset_or_manual_c(self):
        s = Session()
        hand = {"left": np.array([.2, .3, 1.2]), "right": np.array([.2, -.3, 1.2])}
        s.calibrate("right", hand["right"])
        s.run(0, .5, {"left": False, "right": False}, hand)
        s.auto.forget()   # Space, H or C while the gap is open
        s.run(.5, 1., {"left": False, "right": True}, hand)
        self.assertEqual(s.recentered, [])

    def test_calibrate_side_recenter_anchors_at_current_wrist(self):
        from view_dual_arms import advance_physics, calibrate_side, setup
        model, data, controller, _ = setup("human", workspace=True)
        mapper = WristMapper(.6)
        sample = (0., {}, wrist([.2, -.3, 1.2]), None)
        calibrate_side(data, controller, mapper, "right", sample)
        mapper.position_scales["right"][1] = 1.4
        controller.set_target("right", controller.home["right"] + [.05, .03, -.04])
        for _ in range(150):
            controller.command(data)
            advance_physics(model, data, controller)
        mapper.invalidate("right")
        moved = (0., {}, wrist([.3, -.1, 1.1]), None)
        _, ok = calibrate_side(data, controller, mapper, "right", moved, recenter=True)
        self.assertTrue(ok)
        site = data.site_xpos[controller.sites["right"]]
        np.testing.assert_allclose(controller.targets["right"], site, atol=1e-12)
        np.testing.assert_allclose(mapper.target("right", moved[2]), site, atol=1e-12)
        np.testing.assert_allclose(mapper.position_scales["right"], [.6, 1.4, .6])
        # Never calibrated: refused, and nothing is anchored.
        _, ok = calibrate_side(data, controller, mapper, "left", moved, recenter=True)
        self.assertFalse(ok)
        self.assertNotIn("left", mapper.origins)

    def test_default_off_and_argument_checks(self):
        with mock.patch.object(sys, "argv", ["view_dual_arms.py"]):
            args = parse_args()
        self.assertFalse(args.auto_recenter)
        self.assertEqual(args.recenter_window, 3.0)
        for argv in (["--auto-recenter"], ["--address", "x", "--recenter-window", "0"]):
            with mock.patch.object(sys, "argv", ["view_dual_arms.py", *argv]), \
                    self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args()
        with self.assertRaises(ValueError):
            AutoRecenter(0)


if __name__ == "__main__":
    unittest.main()
