"""Offline checks; these do not validate a headset or graphical viewer."""
import unittest
import mujoco
import numpy as np
from arm_control import PALM_DOWN_ROTATION, rotation_matrix, rotation_vector
from view_dual_arms import setup, advance_physics


class HumanArmsTests(unittest.TestCase):
    def setUp(self):
        self.model,self.data,self.controller,self.hands=setup("human")

    def test_structure_and_home(self):
        m,d,c=self.model,self.data,self.controller
        self.assertEqual(m.nu,50)
        self.assertEqual(m.nq-m.nv,4)
        self.assertEqual(d.ncon,0)
        for s in ("left","right"):
            self.assertEqual(len(self.hands[s]),18)
            self.assertEqual(m.joint(f"{s}_shoulder_ball").type[0],mujoco.mjtJoint.mjJNT_BALL)
            self.assertEqual(m.joint(f"{s}_elbow_hinge").type[0],mujoco.mjtJoint.mjJNT_HINGE)
            self.assertEqual(m.joint(f"{s}_wrist_ball").type[0],mujoco.mjtJoint.mjJNT_BALL)
            np.testing.assert_allclose(c.home_rotations[s],PALM_DOWN_ROTATION,atol=1e-9)

    def test_physical_inward_quarter_turn_and_reach(self):
        m,d,c=self.model,self.data,self.controller
        # World +X is the forward arm direction; opposite signs turn palms inward.
        for s,sign in (("left",-1),("right",1)):
            c.set_rotation(s,rotation_matrix([sign*np.pi/2,0,0]) @ c.home_rotations[s])
            c.set_target(s,c.home[s]+[.06,0,.04])
        for _ in range(650):
            c.command(d); advance_physics(m,d,c)
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertLess(np.max(np.abs(d.qvel)),.1)
        for s in ("left","right"):
            self.assertLess(np.linalg.norm(d.site_xpos[c.sites[s]]-c.targets[s]),.006)
            actual=d.site_xmat[c.sites[s]].reshape(3,3)
            self.assertLess(np.linalg.norm(rotation_vector(c.rotation_targets[s] @ actual.T)),.035)
            self.assertLessEqual(np.linalg.norm(d.ctrl[c.shoulder_a[s]]),18.0001)
            self.assertLessEqual(np.linalg.norm(d.ctrl[c.ball_acts[s]]),4.0001)
        self.assertEqual(d.ncon,0)


if __name__=="__main__":
    unittest.main()
