import unittest

import mujoco
import numpy as np

from hand_retargeting import actuator_indices, apply_targets, finger_targets, synthetic_skeleton
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
