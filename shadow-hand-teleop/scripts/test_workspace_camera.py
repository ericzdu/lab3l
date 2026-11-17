"""Workspace camera framing and recovery from the MuJoCo window's own Reset."""
import unittest

import mujoco
import numpy as np

from arm_control import SIDES, WristMapper, rotation_matrix
from view_dual_arms import (WORKSPACE_OPERATOR_VIEW, advance_physics, configure_camera,
                            recover_native_reset, setup)


def project(model, data, camera, point, aspect=4 / 3):
    """Normalised image coordinates in [-1, 1] (x right, y up) for a world point."""
    scene = mujoco.MjvScene(model, maxgeom=2000)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    cam = scene.camera[0]
    forward, up = np.array(cam.forward), np.array(cam.up)
    right = np.cross(forward, up)
    rel = np.asarray(point) - np.array(cam.pos)
    depth = rel @ forward
    half = np.tan(np.deg2rad(model.vis.global_.fovy) / 2)
    return np.array([(rel @ right) / (depth * half * aspect), (rel @ up) / (depth * half)]), depth


class WorkspaceCameraTests(unittest.TestCase):
    def setUp(self):
        self.model, self.data, self.controller, _ = setup("human", workspace=True)
        mujoco.mj_forward(self.model, self.data)

    def test_both_views_frame_hands_cup_and_tray(self):
        points = {"cup": self.data.body("cup").xpos, "tray": self.data.body("tray").xpos}
        for side in SIDES:
            points[side + " palm"] = self.data.body(f"{side}_{side[0]}h_palm").xpos
        for overview in (False, True):
            camera = mujoco.MjvCamera()
            configure_camera(camera, overview=overview, workspace=True)
            for name, point in points.items():
                xy, depth = project(self.model, self.data, camera, point)
                self.assertGreater(depth, 0, f"{name} behind camera (overview={overview})")
                self.assertTrue(np.all(np.abs(xy) < .9), f"{name} outside frame (overview={overview}): {xy}")

    def test_operator_view_keeps_left_on_left(self):
        camera = mujoco.MjvCamera()
        configure_camera(camera, workspace=True)
        left = project(self.model, self.data, camera, self.data.body("left_lh_palm").xpos)[0]
        right = project(self.model, self.data, camera, self.data.body("right_rh_palm").xpos)[0]
        self.assertLess(left[0], right[0])
        self.assertEqual(WORKSPACE_OPERATOR_VIEW["azimuth"], 0)

    def test_other_variants_keep_their_camera(self):
        camera = mujoco.MjvCamera()
        configure_camera(camera)
        np.testing.assert_allclose(camera.lookat, [0.55, 0, 0.62])
        self.assertEqual((camera.distance, camera.elevation), (0.95, -32))


class NativeResetTests(unittest.TestCase):
    def test_native_reset_is_recovered_to_home(self):
        model, data, controller, _ = setup("human", workspace=True)
        mapper = WristMapper()
        for side, sign in (("left", 1), ("right", -1)):
            controller.set_target(side, controller.home[side] + [.1, -sign * .08, .08])
            controller.set_rotation(side, rotation_matrix([sign, .3, 0]) @ controller.home_rotations[side])
            mapper.calibrate(side, np.eye(4), controller.targets[side])
        for _ in range(200):
            controller.command(data); advance_physics(model, data, controller)
        last = data.time
        mujoco.mj_resetData(model, data)  # what the viewer's Reset button does
        self.assertFalse(recover_native_reset(data, controller, mapper, 0.))
        self.assertTrue(recover_native_reset(data, controller, mapper, last))
        self.assertEqual(mapper.origins, {})
        for _ in range(100):
            controller.command(data, active=[]); advance_physics(model, data, controller)
        for side in SIDES:
            self.assertLess(np.linalg.norm(data.site_xpos[controller.sites[side]] - controller.home[side]), .005)
            # Hands stay above the worktop, not dropped through it.
            self.assertGreater(data.body(f"{side}_{side[0]}h_palm").xpos[2], .62)
        self.assertLess(np.max(np.abs(data.qvel)), .5)


if __name__ == "__main__":
    unittest.main()
