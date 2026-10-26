"""Physics checks for props; not an end-to-end teleoperation grasp test."""
import unittest
import mujoco
import numpy as np
from view_dual_arms import setup, advance_physics
from workspace_scene import CUP_START, TABLE_TOP, TRAY_CENTER


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.m,self.d,self.c,_=setup("human",workspace=True)
        self.q=int(self.m.joint("cup_free").qposadr[0])

    def step(self,n=300):
        for _ in range(n):
            self.c.command(self.d)
            advance_physics(self.m,self.d,self.c)

    def test_stable_table_and_reset(self):
        self.assertEqual(self.m.body_gravcomp[self.m.body("cup").id],0)
        self.assertAlmostEqual(self.m.body("cup").mass[0],.21)
        self.step()
        np.testing.assert_allclose(self.d.qpos[self.q:self.q+3],CUP_START,atol=.002)
        self.assertLess(np.max(np.abs(self.d.qvel)),.01)
        self.d.qpos[self.q]+= .08
        self.c.reset(self.d)
        np.testing.assert_allclose(self.d.qpos[self.q:self.q+3],CUP_START)

    def test_gravity_and_tray_support(self):
        self.d.qpos[self.q:self.q+3]=[*TRAY_CENTER,TABLE_TOP+.20]
        mujoco.mj_forward(self.m,self.d)
        initial=self.d.qpos[self.q+2]
        self.step(10)
        self.assertLess(self.d.qpos[self.q+2],initial-.01)
        self.step()
        np.testing.assert_allclose(self.d.qpos[self.q:self.q+2],TRAY_CENTER,atol=.003)
        self.assertAlmostEqual(self.d.qpos[self.q+2],TABLE_TOP+.012+.057,delta=.003)
        self.assertLess(np.max(np.abs(self.d.qvel)),.02)

    def test_open_cup_collision_geometry(self):
        # There is no solid cylinder filling the cavity or closing its mouth.
        walls=[self.m.geom(f"cup_wall_{i}").id for i in range(24)]
        self.assertTrue(all(self.m.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX for i in walls))
        self.assertEqual(self.m.joint("cup_free").type[0],mujoco.mjtJoint.mjJNT_FREE)

    def test_named_constants_preserve_geometry(self):
        # Literal values from before the constants refactor (labeler depends on them).
        expected={
            "tray_base":([.10,.075,.006],[0,0,.006]),
            "tray_rim_0_-1":([.005,.075,.016],[-.10,0,.022]),
            "tray_rim_0_1":([.005,.075,.016],[.10,0,.022]),
            "tray_rim_1_-1":([.10,.005,.016],[0,-.075,.022]),
            "tray_rim_1_1":([.10,.005,.016],[0,.075,.022]),
            "cup_bottom":([.034,.006,0],[0,0,-.051]),
        }
        for name,(size,pos) in expected.items():
            geom=self.m.geom(name)
            np.testing.assert_allclose(geom.size,size,atol=1e-12,err_msg=name)
            np.testing.assert_allclose(geom.pos,pos,atol=1e-12,err_msg=name)
        np.testing.assert_allclose(self.m.body("tray").pos,[.43,-.12,.505],atol=1e-12)
        np.testing.assert_allclose(CUP_START,[.32,-.28,.562],atol=1e-12)
        from workspace_scene import CUP_HALF_HEIGHT,CUP_RADIUS,TRAY_HALF_EXTENTS,TRAY_RIM_THICKNESS
        self.assertEqual((TRAY_HALF_EXTENTS,TRAY_RIM_THICKNESS,CUP_RADIUS,CUP_HALF_HEIGHT),
                         ((.10,.075),.005,.034,.057))


if __name__=="__main__":
    unittest.main()
