"""--clean-ui / --no-panel flags, goal-marker hiding and throttled status lines."""
import contextlib
import io
import sys
import unittest
from unittest import mock

import numpy as np

from arm_control import SIDES
from view_dual_arms import StatusPrinter, hide_goal_markers, parse_args, setup


class CleanUiTests(unittest.TestCase):
    def test_flags_default_off(self):
        with mock.patch.object(sys, "argv", ["view_dual_arms.py"]):
            args = parse_args()
        self.assertFalse(args.clean_ui or args.no_panel)
        with mock.patch.object(sys, "argv", ["view_dual_arms.py", "--wrist", "human", "--workspace",
                                             "--address", "x", "--clean-ui", "--no-panel"]):
            args = parse_args()
        self.assertTrue(args.clean_ui and args.no_panel)

    def test_hide_goal_markers_only_changes_alpha(self):
        model, data, _, _ = setup("human", workspace=True)
        sizes = (model.nq, model.nv, model.nu, model.ngeom)
        before = model.geom_rgba.copy()
        hide_goal_markers(model)
        goals = np.isin(model.geom_bodyid, [model.body(f"{side}_goal").id for side in SIDES])
        self.assertEqual(goals.sum(), 2)
        self.assertTrue((model.geom_rgba[goals, 3] == 0).all())
        np.testing.assert_array_equal(model.geom_rgba[~goals], before[~goals])
        np.testing.assert_array_equal(model.geom_rgba[goals, :3], before[goals, :3])
        self.assertEqual((model.nq, model.nv, model.nu, model.ngeom), sizes)


class StatusPrinterTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.printer = StatusPrinter(write=self.lines.append)

    def test_coalesces_flicker_to_once_per_second(self):
        urgent = ((True, True), False, None, False)
        for i in range(100):                          # 1 s of LIVE/WAIT flicker at 100 Hz
            self.printer.update("left: LIVE" if i % 2 else "left: WAIT", i * .01, urgent)
        self.assertEqual(self.lines, ["left: WAIT"])
        self.printer.update("left: LIVE", 1.05, urgent)  # settled state shows within ~1 s
        self.assertEqual(self.lines, ["left: WAIT", "left: LIVE"])
        self.printer.update("left: LIVE", 5., urgent)   # unchanged text: no reprint
        self.assertEqual(len(self.lines), 2)

    def test_urgent_changes_print_immediately(self):
        self.printer.update("a", 0., ((False, False), False, None, False))
        for t, status, urgent in ((.1, "b", ((True, False), False, None, False)),     # FOLLOW
                                  (.2, "c", ((True, False), False, None, True)),      # REC start
                                  (.3, "Connection error: x", ((True, False), False, "x", True))):
            self.printer.update(status, t, urgent)
        self.assertEqual(self.lines, ["a", "b", "c", "Connection error: x"])


if __name__ == "__main__":
    unittest.main()
