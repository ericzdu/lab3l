"""Fast checks of the reachability probe helpers (the full sweep is a CLI)."""
import contextlib
import io
import unittest

import numpy as np

from check_reach import CLIP_HIGH, CLIP_LOW, REACH_CAP, Probe, limit_clearance, orientation, parse_args


class CheckReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = Probe()
        cls.controller = cls.probe.controller

    def test_clearance_matches_set_target(self):
        c = self.controller
        for side in ("left", "right"):
            home = c.home[side]
            shoulder = c.model.body_pos[c.shoulder_b[side]]
            expected = min(np.min(-CLIP_LOW[side]), np.min(CLIP_HIGH[side]),
                           REACH_CAP - np.linalg.norm(home - shoulder))
            self.assertAlmostEqual(limit_clearance(c, side, home), expected, places=9)
            # Beyond the clip box: negative, and equal to set_target's reach error.
            beyond = limit_clearance(c, side, home + [0, 0, -.25])
            self.assertLess(beyond, 0)
            self.assertAlmostEqual(-beyond, c.reach_errors[side], places=9)

    def test_palm_inward_faces_midline(self):
        c = self.controller
        for side, sign in (("left", -1), ("right", 1)):
            # Shadow palms face local -Y; at home that is world -Z (palm down).
            down = orientation(c, side, "palm_down")
            np.testing.assert_allclose(-down[:, 1], [0, 0, -1], atol=1e-9)
            inward = orientation(c, side, "palm_inward")
            np.testing.assert_allclose(-inward[:, 1], [0, sign, 0], atol=1e-9)
            pitched = orientation(c, side, "palm_inward_pitch30")
            # Fingers (local +Z) tip 30 degrees below horizontal.
            self.assertAlmostEqual(pitched[2, 2], -np.sin(np.deg2rad(30)), places=9)

    def test_cup_and_tray_do_not_collide_in_probe(self):
        m = self.probe.model
        for geom in np.r_[self.probe.cup, self.probe.tray]:
            self.assertEqual(m.geom_contype[geom] | m.geom_conaffinity[geom], 0)
        self.assertTrue(len(self.probe.hand["right"]))

    def test_argument_errors(self):
        for argv in (["--step", "0"], ["--settle", "-1"], ["--workers", "-2"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
