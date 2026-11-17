"""Bimanual span fitting and real left/right hand contact, without a headset."""
import unittest
import numpy as np
from arm_control import WristMapper, rotation_matrix
from view_dual_arms import setup,advance_physics


class BimanualTests(unittest.TestCase):
    def test_span_fit_closes_the_artificial_anchor_gap(self):
        mapper=WristMapper(.6)
        wrists={}
        for side,sign in (('left',1),('right',-1)):
            wrist=np.eye(4);wrist[0,3]=-sign*.2
            wrists[side]=wrist
            mapper.calibrate(side,wrist,[.2,sign*.25,.75])
        self.assertTrue(mapper.calibrate_hand_span())
        for side in ('left','right'):
            np.testing.assert_allclose(mapper.target(side,wrists[side]),mapper.anchors[side])
            touching=wrists[side].copy();touching[0,3]=0
            self.assertAlmostEqual(mapper.target(side,touching)[1],0,places=8)
        before=mapper.target('right',np.eye(4)).copy()
        mapper.invalidate('left')
        np.testing.assert_allclose(mapper.target('right',np.eye(4)),before)
        self.assertIsNone(mapper.target('left',np.eye(4)))

    def test_invalid_span_falls_back_without_extreme_gains(self):
        mapper=WristMapper()
        for side in ('left','right'):
            mapper.calibrate(side,np.eye(4),[.2,.25 if side=='left' else -.25,.75])
        self.assertFalse(mapper.calibrate_hand_span())
        for side in ('left','right'):
            np.testing.assert_allclose(mapper.position_scales[side],[.6,.6,.6])

    def test_palms_make_physical_contact_then_separate(self):
        m,d,c,_=setup('human')
        for side,sign in (('left',1),('right',-1)):
            target=c.home[side].copy();target[1]=sign*.012
            c.set_target(side,target)
            self.assertLess(abs(c.targets[side][1]),.02)
            c.set_rotation(side,rotation_matrix([-sign*np.pi/2,0,0])@c.home_rotations[side])
        for _ in range(650):
            c.command(d);advance_physics(m,d,c)
        contacts=[]
        for contact in d.contact:
            bodies=[m.body(m.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            if set(bodies)=={'left_lh_palm','right_rh_palm'}:
                contacts.append(contact.dist)
        self.assertTrue(contacts,'Must be palm contact, not only bulky wrist housings')
        self.assertGreater(min(contacts),-.001)
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertLess(np.max(abs(d.qvel)),.05)
        for side,sign in (('left',1),('right',-1)):
            self.assertGreater(sign*d.body(f'{side}_elbow').xpos[1],.08)
            c.set_target(side,c.home[side])
        for _ in range(400):
            c.command(d);advance_physics(m,d,c)
        self.assertEqual(d.ncon,0)
        for side in ('left','right'):
            self.assertLess(np.linalg.norm(d.site_xpos[c.sites[side]]-c.home[side]),.003)


if __name__=='__main__':
    unittest.main()
