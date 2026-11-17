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

    def push(self,force,height,seconds=.3):
        """Horizontal +x force at `height` above the cup centre, then let it settle."""
        cup=self.m.body("cup").id
        for _ in range(int(seconds/.01)):
            self.d.qfrc_applied[:]=0
            point=self.d.xpos[cup]+[0,0,height]
            mujoco.mj_applyFT(self.m,self.d,[force,0,0],[0,0,0],point,cup,self.d.qfrc_applied)
            self.step(1)
        self.d.qfrc_applied[:]=0
        self.step(150)

    def tilt(self):
        return np.degrees(np.arccos(np.clip(self.d.body("cup").xmat[8],-1,1)))

    def test_cup_slides_when_pushed_and_tips_when_knocked(self):
        # Contact condim 4 (torsional friction) must not glue the cup down.
        self.assertEqual(self.m.geom("cup_wall_0").condim[0],4)
        self.step(100)
        start=self.d.body("cup").xpos.copy()
        self.push(2.5,-.045)                       # low push, above the ~2.1 N friction limit
        self.assertGreater(self.d.body("cup").xpos[0]-start[0],.01)
        self.assertLess(self.tilt(),10)
        self.c.reset(self.d)
        self.step(100)
        self.push(2.,.057,.1)                      # short knock at the rim
        self.assertGreater(self.tilt(),60)
        self.assertGreater(self.d.body("cup").xpos[2],TABLE_TOP)   # fell over on the table

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

    def test_scene_version_2_constants(self):
        from workspace_scene import SCENE_VERSION,SCENE_V1
        self.assertEqual(SCENE_VERSION,2)
        self.assertEqual(TABLE_TOP,.60)
        np.testing.assert_allclose(CUP_START,[.30,-.27,.657],atol=1e-12)
        np.testing.assert_allclose(TRAY_CENTER,[.30,-.09],atol=1e-12)
        self.assertEqual(SCENE_V1,{"table_top":.505,"cup_start":(.32,-.28,.562),"tray_center":(.43,-.12)})

    def test_geometry_follows_constants(self):
        # Sizes are unchanged since version 1; positions follow the constants.
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
        np.testing.assert_allclose(self.m.body("tray").pos,[*TRAY_CENTER,TABLE_TOP],atol=1e-12)
        np.testing.assert_allclose(self.m.body("cup").pos,CUP_START,atol=1e-12)
        from workspace_scene import CUP_HALF_HEIGHT,CUP_RADIUS,TRAY_HALF_EXTENTS,TRAY_RIM_THICKNESS
        self.assertEqual((TRAY_HALF_EXTENTS,TRAY_RIM_THICKNESS,CUP_RADIUS,CUP_HALF_HEIGHT),
                         ((.10,.075),.005,.034,.057))
        self.assertAlmostEqual(CUP_START[2],TABLE_TOP+CUP_HALF_HEIGHT,places=12)
        # Legs run from the floor plane to the worktop centre.
        floor=self.m.geom("floor").pos[2]
        worktop=self.m.geom("worktop")
        legs=[self.m.geom(i) for i in range(self.m.ngeom) if self.m.geom(i).name.startswith("table_leg_")]
        self.assertEqual(len(legs),4)
        for leg in legs:
            self.assertAlmostEqual(leg.pos[2]-leg.size[2],floor,places=12)
            self.assertAlmostEqual(leg.pos[2]+leg.size[2],worktop.pos[2],places=12)

    def test_layout_constraints(self):
        from workspace_scene import CUP_RADIUS,TRAY_HALF_EXTENTS
        # Cup at least 6 cm from the tray's outer rim; cup right, tray between hands.
        gap=np.abs(CUP_START[:2]-TRAY_CENTER)-np.asarray(TRAY_HALF_EXTENTS)-CUP_RADIUS
        self.assertGreaterEqual(gap.max(),.06)
        self.assertLess(CUP_START[1],0)
        self.assertTrue(self.c.home["right"][1]<TRAY_CENTER[1]<self.c.home["left"][1])
        self.assertLess(CUP_START[1],TRAY_CENTER[1])
        # At home the cup touches only the table: no arm or hand contact.
        self.step(100)
        cup=self.m.body("cup").id
        for i in range(self.d.ncon):
            bodies={int(self.m.geom_bodyid[self.d.contact.geom1[i]]),int(self.m.geom_bodyid[self.d.contact.geom2[i]])}
            if cup in bodies:
                other=self.m.body((bodies-{cup}).pop() if len(bodies)>1 else cup).name
                self.assertEqual(other,"world",other)

    def test_worktop_follows_table_top(self):
        worktop=self.m.geom("worktop")
        self.assertAlmostEqual(worktop.pos[2]+worktop.size[2],TABLE_TOP,places=12)
        from human_arms import build_human_model
        bare=build_human_model(workspace=False).geom("worktop")
        self.assertAlmostEqual(bare.pos[2]+bare.size[2],TABLE_TOP,places=12)

    def test_other_variants_keep_their_table(self):
        # Panda and spherical-wrist scenes are not driven by TABLE_TOP.
        from arm_control import build_arm_model
        from spherical_wrist import build_spherical_model
        for model in (build_arm_model(),build_spherical_model()):
            worktop=model.geom("worktop")
            np.testing.assert_allclose(worktop.pos,[.62,0,.48],atol=1e-12)
            np.testing.assert_allclose(worktop.size,[.46,.58,.025],atol=1e-12)


if __name__=="__main__":
    unittest.main()
