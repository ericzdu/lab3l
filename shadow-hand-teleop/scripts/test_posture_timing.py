"""Posture-envelope and render-independent timing regressions."""
import unittest
import mujoco
import numpy as np
from view_dual_arms import setup, advance_physics, PhysicsClock
from arm_control import rotation_matrix, rotation_vector


class PostureTimingTests(unittest.TestCase):
    def test_clock_at_different_render_rates(self):
        for fps in (30,60,100,144):
            clock=PhysicsClock(.002)
            count=sum(clock.steps(1/fps) for _ in range(fps))
            self.assertEqual(count,500)
        self.assertLessEqual(PhysicsClock(.002).steps(5),25)
        clock=PhysicsClock(.002)
        clock.steps(.001)
        self.assertEqual(clock.steps(5,running=False),0)
        self.assertEqual(clock.steps(.001),0)
        for elapsed in (-1,np.nan):
            with self.assertRaises(ValueError):
                clock.steps(elapsed)

    def test_mirrored_relaxed_home(self):
        m,d,c,_=setup('human')
        for side,sign in (('left',1),('right',-1)):
            elbow=d.body(f'{side}_elbow').xpos
            shoulder=d.body(f'{side}_shoulder').xpos
            self.assertGreater(sign*(elbow[1]-shoulder[1]),.06)
            self.assertLess(elbow[2],shoulder[2]-.2)
        np.testing.assert_allclose(d.body('left_elbow').xpos*[1,-1,1],d.body('right_elbow').xpos,atol=1e-9)

    def test_inward_guides_cannot_select_inward_elbow(self):
        m,d,c,_=setup('human')
        rng=np.random.default_rng(7)
        for side,sign in (('left',1),('right',-1)):
            for _ in range(80):
                c.set_target(side,c.home[side]+rng.uniform([-.12,-.14,-.16],[.16,.14,.18]))
                pose=c.limb_pose(side,c.targets[side],rng.normal(size=3))
                scratch=mujoco.MjData(m)
                scratch.qpos[:]=d.qpos
                scratch.qpos[c.shoulder_q[side]]=pose[:4]
                scratch.qpos[c.elbow_q[side]]=pose[4]
                mujoco.mj_forward(m,scratch)
                elbow=scratch.body(f'{side}_elbow').xpos
                shoulder=scratch.body(f'{side}_shoulder').xpos
                self.assertGreaterEqual(sign*(elbow[1]-shoulder[1]),.015-1e-6)
                self.assertLess(np.linalg.norm(scratch.site_xpos[c.sites[side]]-c.targets[side]),1e-8)
                self.assertLess(2*np.arccos(abs(pose[0])),2.3)

    def test_physical_response_with_30hz_render_budget(self):
        m,d,c,_=setup('human')
        side='right'
        c.set_target(side,c.home[side]+[.06,0,.04])
        c.set_rotation(side,rotation_matrix([1,0,0])@c.home_rotations[side])
        clock=PhysicsClock(m.opt.timestep)
        for _ in range(36):
            steps=clock.steps(1/30)
            c.command(d,dt=steps*m.opt.timestep)
            advance_physics(m,d,c,steps)
        self.assertAlmostEqual(d.time,1.2,places=8)
        self.assertLess(np.linalg.norm(d.site_xpos[c.sites[side]]-c.targets[side]),.004)
        actual=d.site_xmat[c.sites[side]].reshape(3,3)
        self.assertLess(np.linalg.norm(rotation_vector(c.rotation_targets[side]@actual.T)),.04)
        elbow=d.body(f'{side}_elbow').xpos
        self.assertLess(elbow[1],d.body(f'{side}_shoulder').xpos[1]-.01)
        self.assertTrue(np.isfinite(d.qpos).all())

    def test_physical_elbows_stay_outward_under_inward_guides(self):
        m,d,c,_=setup('human')
        rng=np.random.default_rng(9)
        for frame in range(600):
            if frame%60==0:
                for side,sign in (('left',1),('right',-1)):
                    c.set_target(side,c.home[side]+rng.uniform([-.06,-.08,-.08],[.08,.08,.08]))
                    # Forearm pointing outward would put its elbow inward.
                    c.set_forearm_target(side,[.3,sign,0])
            c.command(d)
            advance_physics(m,d,c)
            for side,sign in (('left',1),('right',-1)):
                margin=sign*(d.body(f'{side}_elbow').xpos[1]-d.body(f'{side}_shoulder').xpos[1])
                # Physical servos may briefly undershoot the 15 mm target.
                self.assertGreater(margin,.005)
        self.assertTrue(np.isfinite(d.qpos).all())


if __name__=='__main__':
    unittest.main()
