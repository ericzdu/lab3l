"""Physical integration checks for the simplified ball-joint assembly."""

import unittest

import mujoco
import numpy as np

from arm_control import SIDES, rotation_matrix, rotation_vector
from hand_retargeting import actuator_indices
from spherical_wrist import SphericalArmController, build_spherical_model, quaternion
from view_dual_arms import advance_physics, set_fist


class SphericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_spherical_model()

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.controller = SphericalArmController(self.model, self.data)

    def test_real_ball_joint_replaces_industrial_wrist(self):
        self.assertEqual(self.data.ncon, 0)
        self.assertEqual(self.model.nu, 52)
        self.assertEqual(self.model.nq - self.model.nv, 2)  # One quaternion per hand.
        for side in SIDES:
            joint = self.model.joint(f"{side}_arm_wrist_ball").id
            self.assertEqual(self.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_BALL)
            self.assertEqual(len(actuator_indices(self.model, side, include_wrist=False)), 18)
            for name in (f"{side}_arm_link6", f"{side}_arm_link7"):
                self.assertEqual(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name), -1)
            for name in ("WRJ1", "WRJ2"):
                self.assertEqual(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                 f"{side}_{side[0]}h_{name}"), -1)
            normal = -self.data.site_xmat[self.controller.sites[side]].reshape(3, 3)[:, 1]
            self.assertLess(normal[2], -.99)

    def test_ball_rotation_has_fixed_pivot(self):
        pivot = self.data.site_xpos[self.controller.sites["left"]].copy()
        right = self.data.qpos.copy()
        qids = self.controller.ball_qids["left"]
        self.data.qpos[qids] = quaternion(rotation_matrix([.3, -.2, .15]))
        mujoco.mj_forward(self.model, self.data)
        np.testing.assert_allclose(self.data.site_xpos[self.controller.sites["left"]], pivot, atol=1e-10)
        np.testing.assert_array_equal(self.data.qpos[self.controller.ball_qids["right"]],
                                      right[self.controller.ball_qids["right"]])

    def test_physical_pose_tracking_and_finger_independence(self):
        right_arm_before = self.data.qpos[self.controller.qids["right"]].copy()
        self.controller.set_target("left", self.controller.home["left"] + [.025, .02, .03])
        for side, vector in (("left", [.2, -.15, .1]), ("right", [-.35, .1, -.12])):
            self.controller.set_rotation(side, rotation_matrix(vector) @ self.controller.home_rotations[side])
            set_fist(self.model, self.data, actuator_indices(self.model, side, include_wrist=False), True)
        for _ in range(350):
            self.controller.command(self.data)
            advance_physics(self.model, self.data, self.controller)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertLess(np.max(np.abs(self.data.qvel)), .05)
        for side in SIDES:
            actual = self.data.site_xmat[self.controller.sites[side]].reshape(3, 3)
            angle = np.linalg.norm(rotation_vector(self.controller.rotation_targets[side] @ actual.T))
            self.assertLess(angle, .02)
            self.assertLess(np.linalg.norm(self.data.site_xpos[self.controller.sites[side]]
                                           - self.controller.targets[side]), .003)
            self.assertAlmostEqual(np.linalg.norm(self.data.qpos[self.controller.ball_qids[side]]), 1, places=6)
            self.assertLessEqual(np.linalg.norm(self.data.ctrl[self.controller.ball_acts[side]]), 4.000001)
            finger = self.model.joint(f"{side}_{side[0]}h_FFJ3").qposadr[0]
            self.assertGreater(self.data.qpos[finger], 1)
        # Pure right-hand rotation no longer needs to swing the Panda wrist.
        np.testing.assert_allclose(self.data.qpos[self.controller.qids["right"]], right_arm_before, atol=.002)
        # Closing a fist creates ordinary finger-to-finger contacts. No
        # contacts should involve the forearm, ball housing, or worktable.
        for contact in self.data.contact:
            bodies = [self.model.body(self.model.geom_bodyid[g]).name
                      for g in (contact.geom1, contact.geom2)]
            self.assertTrue(any(all(name.startswith(f"{side}_{side[0]}h_") for name in bodies)
                                for side in SIDES), bodies)
            self.assertGreater(contact.dist, -.001)

    def test_torque_limits_and_inactive_side_hold(self):
        self.controller.set_rotation("left", rotation_matrix([0, .8, 0]) @ self.controller.home_rotations["left"])
        self.controller.set_rotation("right", rotation_matrix([.5, 0, 0]) @ self.controller.home_rotations["right"])
        before = self.controller.rotations["right"].copy()
        self.controller.command(self.data, active=("left",))
        np.testing.assert_array_equal(self.controller.rotations["right"], before)
        self.data.qvel[self.controller.ball_dofs["left"]] = [100, -100, 100]
        self.controller.update_torques(self.data)
        for side in SIDES:
            self.assertLessEqual(np.linalg.norm(self.data.ctrl[self.controller.ball_acts[side]]), 4.000001)


if __name__ == "__main__":
    unittest.main()
