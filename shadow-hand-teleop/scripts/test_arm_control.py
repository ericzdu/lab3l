"""Offline integration checks: attachment, physical tracking, limits, SDK mapping."""

import time
import unittest

import mujoco
import numpy as np

from arm_control import (ArmController, SIDES, WristMapper, build_arm_model,
                         wrist_position, rotation_matrix, rotation_vector,
                         TRACKING_TO_ROBOT)
from hand_retargeting import actuator_indices, apply_targets, finger_targets, synthetic_skeleton
from teleop_shadow_hands import TrackingBridge
from view_dual_arms import configure_camera, set_fist


class ArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_arm_model()

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.controller = ArmController(self.model, self.data)

    def test_both_hands_are_descendants_of_arm_flange(self):
        self.assertEqual(self.model.nu, 54)  # 2 * (7 arm + 20 hand actuators)
        for side in SIDES:
            body = self.model.body(f"{side}_{side[0]}h_wrist").id
            flange = self.model.body(f"{side}_arm_attachment").id
            while body and body != flange:
                body = self.model.body_parentid[body]
            self.assertEqual(body, flange)
            self.assertEqual(len(actuator_indices(self.model, side)), 20)

    def test_physics_tracks_3d_targets_and_fingers(self):
        # Opposite translations test both arms independently with all axes.
        for side, sign in (("left", 1), ("right", -1)):
            self.controller.set_target(side, self.controller.home[side] + [0.035, sign * 0.03, 0.04])
            apply_targets(self.data.ctrl, finger_targets(synthetic_skeleton(0.5)),
                          actuator_indices(self.model, side), self.model.actuator_ctrlrange)
        for _ in range(400):
            self.controller.command(self.data)
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertLess(np.max(np.abs(self.data.qvel)), 0.05)
        for side in SIDES:
            error = np.linalg.norm(self.data.site_xpos[self.controller.sites[side]] - self.controller.targets[side])
            self.assertLess(error, 0.003, (side, error))
            rotation = self.data.site_xmat[self.controller.sites[side]].reshape(3, 3)
            np.testing.assert_allclose(rotation, self.controller.rotations[side], atol=0.01)
            finger = self.model.joint(f"{side}_{side[0]}h_FFJ3").qposadr[0]
            self.assertGreater(self.data.qpos[finger], 0.2)
        self.assertEqual(self.data.ncon, 0)

    def test_bounds_rate_and_independence(self):
        self.controller.set_target("left", [100, 100, 100])
        self.assertTrue(np.all(np.abs(self.controller.targets["left"] - self.controller.home["left"]) <= 0.181))
        before = self.data.ctrl.copy()
        command = self.controller.solve(self.data, "left")
        acts = self.controller.actuators["left"]
        self.assertTrue(np.all(np.abs(command - before[acts]) <= 0.00600001))
        limits = self.model.jnt_range[self.controller.joints["left"]]
        self.assertTrue(np.all(command >= limits[:, 0]))
        self.assertTrue(np.all(command <= limits[:, 1]))
        # Solving on scratch state must never teleport either real arm.
        np.testing.assert_array_equal(self.data.ctrl, before)
        np.testing.assert_array_equal(self.data.qpos[self.controller.qids["right"]],
                                      before[self.controller.actuators["right"]])
        with self.assertRaises(ValueError):
            self.controller.set_target("left", [np.nan, 0, 0])

    def test_operator_view_palms_down_and_thumbs_inward(self):
        projections = {}
        camera = mujoco.MjvCamera()
        configure_camera(camera)
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None,
                              camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        eye = scene.camera[0].pos
        forward = scene.camera[0].forward
        screen_right = np.cross(forward, scene.camera[0].up)
        self.assertGreater(forward[0], 0.7)
        self.assertLess(forward[2], -0.3)
        for side, sign in (("left", 1), ("right", -1)):
            prefix = f"{side}_{side[0]}h_"
            palm = self.data.body(prefix + "palm")
            normal = -palm.xmat.reshape(3, 3)[:, 1]
            self.assertLess(normal[2], -0.99)
            thumb_offset = self.data.body(prefix + "thbase").xpos - palm.xpos
            self.assertLess(sign * thumb_offset[1], 0)
            tip = self.data.body(prefix + "ffdistal").xpos.copy()
            # Closing an index knuckle should curl toward the table, not up.
            q = self.model.joint(prefix + "FFJ3").qposadr[0]
            self.data.qpos[q] = 0.5
            mujoco.mj_forward(self.model, self.data)
            self.assertLess(self.data.body(prefix + "ffdistal").xpos[2], tip[2] - 0.015)
            wrist = self.data.site_xpos[self.controller.sites[side]]
            depth = np.dot(wrist - eye, forward)
            self.assertGreater(depth, 0)
            projections[side] = np.dot(wrist - eye, screen_right) / depth
        self.assertLess(projections["left"], projections["right"])

    def test_calibration_axes_and_reacquisition(self):
        mapper = WristMapper(0.6)
        wrist = np.eye(4)[None]
        anchor = self.controller.home["left"]
        self.assertIsNone(mapper.target("left", wrist))
        mapper.calibrate("left", wrist, anchor)
        moved = wrist.copy()
        moved[0, :3, 3] = [0.1, 0.2, 0.3]
        np.testing.assert_allclose(mapper.target("left", moved) - anchor, [0.12, -0.06, 0.18])
        mapper.invalidate("left")
        self.assertIsNone(mapper.target("left", moved))
        for invalid in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)):
            with self.assertRaises(ValueError):
                wrist_position(invalid)

    def test_relative_orientation_calibration_and_axis_conversion(self):
        mapper = WristMapper()
        initial = np.eye(4)[None]
        initial[0, :3, :3] = rotation_matrix([0.2, -0.4, 0.3])
        anchor = self.controller.home_rotations["left"]
        mapper.calibrate("left", initial, self.controller.home["left"], anchor)
        np.testing.assert_allclose(mapper.rotation_target("left", initial), anchor, atol=1e-8)
        world_delta = rotation_matrix([0.15, -0.2, 0.12])
        moved = initial.copy()
        moved[0, :3, :3] = TRACKING_TO_ROBOT.T @ world_delta @ TRACKING_TO_ROBOT @ initial[0, :3, :3]
        np.testing.assert_allclose(mapper.rotation_target("left", moved), world_delta @ anchor, atol=1e-8)
        self.assertIsNone(mapper.rotation_target("right", moved))
        mapper.invalidate("left")
        self.assertIsNone(mapper.rotation_target("left", moved))

    def test_orientation_limits_rate_and_angle_wrap(self):
        side = "left"
        for rotation in (np.zeros((3, 3)), np.diag([1, 1, -1]), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                self.controller.set_rotation(side, rotation)
        self.controller.set_rotation(side, rotation_matrix([np.deg2rad(170), 0, 0])
                                     @ self.controller.home_rotations[side])
        angle = np.linalg.norm(rotation_vector(self.controller.rotation_targets[side]
                                               @ self.controller.home_rotations[side].T))
        self.assertAlmostEqual(angle, np.deg2rad(120))
        previous = self.controller.rotations[side].copy()
        wrist_before = self.data.ctrl[self.controller.wrist_acts[side]].copy()
        self.controller.command(self.data)
        step = np.linalg.norm(rotation_vector(self.controller.rotations[side] @ previous.T))
        self.assertLessEqual(step, 0.00800001)
        self.assertTrue(np.all(np.abs(self.data.ctrl[self.controller.wrist_acts[side]] - wrist_before) <= .00600001))
        a = rotation_matrix([0, 0, np.deg2rad(179.9)])
        b = rotation_matrix([0, 0, np.deg2rad(-179.9)])
        self.assertAlmostEqual(np.linalg.norm(rotation_vector(b @ a.T)), np.deg2rad(0.2))
        np.testing.assert_allclose(rotation_matrix(rotation_vector(a)), a, atol=1e-8)

    def test_wrist_bending_and_forearm_roll_physically_track(self):
        # Left bend uses the Shadow hinge; right axial roll uses Panda joints.
        commands = {"left": [0, .2, 0], "right": [.35, 0, 0]}
        for side, vector in commands.items():
            self.controller.set_rotation(side, rotation_matrix(vector) @ self.controller.home_rotations[side])
        for _ in range(350):
            self.controller.command(self.data)
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
        self.assertAlmostEqual(self.data.qpos[self.controller.wrist_qids["left"]][0], .2, delta=.01)
        np.testing.assert_allclose(self.data.qpos[self.controller.wrist_qids["right"]], 0, atol=.01)
        for side in SIDES:
            actual = self.data.site_xmat[self.controller.sites[side]].reshape(3, 3)
            angle = np.linalg.norm(rotation_vector(self.controller.rotation_targets[side] @ actual.T))
            self.assertLess(angle, .01)
            self.assertLess(np.linalg.norm(self.data.site_xpos[self.controller.sites[side]]
                                           - self.controller.targets[side]), .003)
            limits = self.controller.wrist_limits[side]
            self.assertTrue(np.all(self.data.ctrl[self.controller.wrist_acts[side]] >= limits[:, 0]))
            self.assertTrue(np.all(self.data.ctrl[self.controller.wrist_acts[side]] <= limits[:, 1]))
            before = self.data.ctrl[self.controller.wrist_acts[side]].copy()
            set_fist(self.model, self.data, actuator_indices(self.model, side), True)
            np.testing.assert_array_equal(self.data.ctrl[self.controller.wrist_acts[side]], before)
        self.assertEqual(self.data.ncon, 0)

    def test_stream_bridge_copies_wrists_and_does_not_refresh_cached_packet(self):
        class Packet(dict):
            pass

        sample = Packet({f"{side}_fingers": synthetic_skeleton(0.2) for side in SIDES})
        sample.update({f"{side}_wrist": np.eye(4)[None] for side in SIDES})
        sample._raw = object()

        class FakeStreamer:
            def __init__(self, **kwargs):
                self.closed = False

            def get_latest(self):
                return sample

            def cleanup(self):
                self.closed = True

        bridge = TrackingBridge(FakeStreamer, "offline-test", with_wrist=True)
        bridge.thread.start()
        try:
            deadline = time.monotonic() + 2
            packets, error = bridge.snapshot()
            while not packets and time.monotonic() < deadline:
                bridge.stop.wait(0.01)
                packets, error = bridge.snapshot()
            self.assertIsNone(error)
            self.assertEqual(set(packets), set(SIDES))
            timestamp = packets["left"][0]
            sample["left_wrist"][0, 0, 3] = 99
            np.testing.assert_allclose(packets["left"][2], np.eye(4)[None])
            bridge.stop.wait(0.03)
            self.assertEqual(bridge.snapshot()[0]["left"][0], timestamp)
        finally:
            bridge.close()
        self.assertTrue(bridge.streamer.closed)


if __name__ == "__main__":
    unittest.main()
