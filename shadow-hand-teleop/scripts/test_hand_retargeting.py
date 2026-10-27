import unittest

import mujoco
import numpy as np

from hand_retargeting import (FIST_THJ4, THJ4_MAX, actuator_indices, apply_targets, finger_targets,
                              synthetic_grasp_skeleton, synthetic_skeleton, thumb_opposition)
from view_dual_shadow_hands import build_model


class RetargetingTests(unittest.TestCase):
    def test_straight_and_bent(self):
        straight = finger_targets(synthetic_skeleton())
        bent = finger_targets(synthetic_skeleton(0.6))
        self.assertAlmostEqual(straight["FFJ0"], 0)
        self.assertAlmostEqual(bent["FFJ3"], 0.6)
        self.assertAlmostEqual(bent["FFJ0"], 1.2)
        self.assertGreater(bent["THJ1"], straight["THJ1"])

    def test_independent_fingers(self):
        joints = synthetic_skeleton()
        joints[5:10] = synthetic_skeleton(0.7)[5:10]
        values = finger_targets(joints)
        self.assertGreater(values["FFJ0"], 1)
        self.assertAlmostEqual(values["MFJ0"], 0)

    def test_pose_scale_and_mirror_invariance(self):
        joints = synthetic_skeleton(0.6)
        moved = joints.copy()
        moved[:, :3, 3] = joints[:, :3, 3] @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) * 2 + 3
        np.testing.assert_allclose(list(finger_targets(joints).values()), list(finger_targets(moved).values()), atol=1e-7)

    def test_open_hand_has_low_opposition(self):
        values = finger_targets(synthetic_grasp_skeleton(0.))
        self.assertLess(values["THJ4"], .1)
        self.assertEqual((values["THJ5"], values["THJ3"]), (0., 0.))

    def test_thumb_across_is_opposed(self):
        fist = finger_targets(synthetic_grasp_skeleton(1.))
        self.assertGreaterEqual(fist["THJ4"], FIST_THJ4)
        self.assertLessEqual(fist["THJ4"], THJ4_MAX)
        # Pinch: fingers mostly open, thumb swung across: opposition from tracking alone.
        pinch = synthetic_grasp_skeleton(0.)
        pinch[1:5] = synthetic_grasp_skeleton(1.)[1:5]
        self.assertGreater(thumb_opposition(pinch[:, :3, 3]), .8)
        self.assertGreater(finger_targets(pinch)["THJ4"], .9)
        values = [finger_targets(synthetic_grasp_skeleton(c))["THJ4"] for c in np.linspace(0, 1, 6)]
        self.assertTrue(np.all(np.diff(values) >= 0), values)

    def test_grip_coupling_raises_thumb_with_flat_thumb(self):
        # Fist with the thumb left in the palm plane: coupling still brings THJ4 to the fist value.
        joints = synthetic_grasp_skeleton(1.)
        joints[1:5] = synthetic_grasp_skeleton(0.)[1:5]
        self.assertLess(thumb_opposition(joints[:, :3, 3]), .1)
        self.assertAlmostEqual(finger_targets(joints)["THJ4"], FIST_THJ4)

    def test_opposition_invariant_to_rotation_scale_and_mirror(self):
        rng = np.random.default_rng(3)
        for closure in (0., .4, .7, 1.):
            joints = synthetic_grasp_skeleton(closure)
            expected = finger_targets(joints)
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            for transform in (q, -q, np.diag([-1., 1, 1])):   # rotations and mirror images
                moved = joints.copy()
                moved[:, :3, 3] = joints[:, :3, 3] @ transform.T * 1.7 + rng.normal(size=3)
                np.testing.assert_allclose(list(finger_targets(moved).values()), list(expected.values()),
                                           atol=1e-7)

    def test_invalid_tracking(self):
        for joints in (np.zeros((25, 4, 4)), np.full((25, 4, 4), np.nan), np.zeros((3, 4, 4))):
            with self.assertRaises(ValueError):
                finger_targets(joints)

    def test_two_hand_physics_and_limits(self):
        model = build_model()
        data = mujoco.MjData(model)
        for side in ("left", "right"):
            apply_targets(data.ctrl, finger_targets(synthetic_skeleton(1.4)), actuator_indices(model, side), model.actuator_ctrlrange)
        self.assertTrue((data.ctrl >= model.actuator_ctrlrange[:, 0]).all())
        self.assertTrue((data.ctrl <= model.actuator_ctrlrange[:, 1]).all())
        for _ in range(500):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(np.linalg.norm(data.qpos), 0.1)


if __name__ == "__main__":
    unittest.main()
