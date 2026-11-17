"""Offline command smoothing, without claiming a validated cup-release pose."""
import unittest
import numpy as np
from view_dual_arms import setup, set_fist, FingerTransition, PhysicsClock


class HandTransitionTests(unittest.TestCase):
    def setUp(self):
        self.m,self.d,self.c,self.hands=setup('human',workspace=True)
        self.motion=FingerTransition(self.m)

    def test_open_is_gradual_and_never_teleports_state(self):
        hand=self.hands['right']
        set_fist(self.m,self.d,hand,True)
        before=self.d.ctrl.copy();qpos=self.d.qpos.copy()
        self.motion.request(self.d,hand,False)
        np.testing.assert_array_equal(self.d.ctrl,before)
        self.motion.advance(self.d,.65)
        ids=list(hand.values())
        np.testing.assert_allclose(self.d.ctrl[ids],before[ids]*.5)
        untouched=np.setdiff1d(np.arange(self.m.nu),ids)
        np.testing.assert_array_equal(self.d.ctrl[untouched],before[untouched])
        np.testing.assert_array_equal(self.d.qpos,qpos)
        self.motion.advance(self.d,.65)
        np.testing.assert_allclose(self.d.ctrl[ids],0,atol=1e-10)
        self.assertFalse(self.motion.tasks)

    def test_repeated_keys_do_not_restart_and_reversal_has_no_jump(self):
        hand=self.hands['right']
        set_fist(self.m,self.d,hand,True)
        for _ in range(13):
            self.motion.request(self.d,hand,False)
            self.motion.advance(self.d,.1)
        np.testing.assert_allclose(self.d.ctrl[list(hand.values())],0,atol=1e-10)
        self.motion.request(self.d,hand,True)
        self.motion.advance(self.d,.3)
        before=self.d.ctrl.copy()
        self.motion.request(self.d,hand,False)
        np.testing.assert_array_equal(self.d.ctrl,before)
        self.motion.advance(self.d,1.3)
        np.testing.assert_allclose(self.d.ctrl[list(hand.values())],0,atol=1e-10)

    def test_render_rate_does_not_change_release_duration(self):
        hand=self.hands['right']
        for fps in (30,60,100):
            set_fist(self.m,self.d,hand,True)
            motion=FingerTransition(self.m)
            motion.request(self.d,hand,False)
            clock=PhysicsClock(self.m.opt.timestep)
            for _ in range(int(1.3*fps)):
                for _ in range(clock.steps(1/fps)):
                    motion.advance(self.d,self.m.opt.timestep)
            np.testing.assert_allclose(self.d.ctrl[list(hand.values())],0,atol=1e-8)

    def test_clear_and_duration_validation(self):
        self.motion.request(self.d,self.hands['left'],True)
        self.motion.clear()
        before=self.d.ctrl.copy()
        self.motion.advance(self.d,.1)
        np.testing.assert_array_equal(self.d.ctrl,before)
        for value in (0,-1,np.nan):
            with self.assertRaises(ValueError):
                FingerTransition(self.m,open_seconds=value)

    def test_reach_warning_without_changing_base_or_arm_state(self):
        base=self.m.body_pos.copy();qpos=self.d.qpos.copy()
        self.c.set_target('right',[100,-100,100])
        self.assertGreater(self.c.reach_errors['right'],1)
        np.testing.assert_array_equal(self.m.body_pos,base)
        np.testing.assert_array_equal(self.d.qpos,qpos)
        self.c.set_target('right',self.c.home['right'])
        self.assertAlmostEqual(self.c.reach_errors['right'],0)
        self.c.reset(self.d)
        self.assertTrue(all(value==0 for value in self.c.reach_errors.values()))


if __name__=='__main__':
    unittest.main()
