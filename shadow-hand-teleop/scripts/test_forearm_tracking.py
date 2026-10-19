"""Synthetic packets and offline physics only; no headset connection."""
import time
import unittest
import numpy as np
from arm_control import WristMapper, TRACKING_TO_ROBOT, rotation_matrix, rotation_vector
from forearm_tracking import forearm_vector
from hand_retargeting import synthetic_skeleton
from teleop_shadow_hands import TrackingBridge
from view_dual_arms import setup, advance_physics


def skeleton27():
    arm=np.repeat(np.eye(4)[None],27,axis=0)
    arm[:25]=synthetic_skeleton(.2)
    arm[26,:3,3]=[-.25,0,0]
    return arm


class ForearmTests(unittest.TestCase):
    def test_wrist_local_points_converted_to_world_vector(self):
        wrist=np.eye(4)
        wrist[:3,:3]=rotation_matrix([0,0,np.pi/2])
        wrist[:3,3]=[10,20,30]  # Translation must cancel out of a direction.
        np.testing.assert_allclose(forearm_vector(skeleton27(),wrist),[0,.25,0],atol=1e-10)

    def test_optional_data_rejects_missing_invalid_and_placeholder_points(self):
        for arm in (None,synthetic_skeleton(),np.zeros((27,4,4)),np.ones(3)):
            self.assertIsNone(forearm_vector(arm,np.eye(4)))
        for length in (0,.01,2,np.nan):
            arm=skeleton27();arm[26,0,3]=-length
            self.assertIsNone(forearm_vector(arm,np.eye(4)))

    def test_relative_direction_calibration_and_invalidation(self):
        mapper=WristMapper();wrist=np.eye(4)
        source=np.array([.25,0,0]);robot=np.array([1.,0,0])
        self.assertIsNone(mapper.forearm_target('left',source))
        mapper.calibrate('left',wrist,[0,0,0],np.eye(3),source,robot)
        np.testing.assert_allclose(mapper.forearm_target('left',source),robot,atol=1e-8)
        moved=rotation_matrix([0,0,.2])@source
        expected=mapper.forearm_alignments['left']@TRACKING_TO_ROBOT@(moved/.25)
        np.testing.assert_allclose(mapper.forearm_target('left',moved),expected)
        self.assertIsNone(mapper.forearm_target('left',None))
        mapper.invalidate('left')
        self.assertIsNone(mapper.forearm_target('left',source))
        # Antiparallel calibration is defined, without a zero cross-product axis.
        robot=-(TRACKING_TO_ROBOT@source)/.25
        mapper.calibrate('left',wrist,[0,0,0],np.eye(3),source,robot)
        np.testing.assert_allclose(mapper.forearm_target('left',source),robot,atol=1e-8)
        mapper.calibrate('left',wrist,[0,0,0],np.eye(3))
        self.assertNotIn('left',mapper.forearm_alignments)

    def test_bridge_preserves_fingers_when_forearm_absent_or_invalid(self):
        for arm in (None, np.zeros((27,4,4)), skeleton27()):
            class Packet(dict):
                pass
            packet=Packet(left_fingers=synthetic_skeleton(),left_wrist=np.eye(4)[None])
            if arm is not None:
                packet['left_arm']=arm
            packet._raw=object()
            class FakeStreamer:
                def __init__(self,**kwargs):
                    pass
                def get_latest(self):
                    return packet
                def cleanup(self):
                    pass
            bridge=TrackingBridge(FakeStreamer,'test',with_wrist=True)
            bridge.thread.start()
            try:
                deadline=time.monotonic()+2
                while not bridge.snapshot()[0] and time.monotonic()<deadline:
                    bridge.stop.wait(.01)
                samples,error=bridge.snapshot()
                self.assertIsNone(error)
                self.assertIn('left',samples)
                self.assertIn('FFJ3',samples['left'][1])
                if arm is not None and np.any(arm):
                    vector=samples['left'][3]
                    np.testing.assert_allclose(vector,[.25,0,0])
                    arm[26,0,3]=-99
                    np.testing.assert_allclose(vector,[.25,0,0])
                else:
                    self.assertIsNone(samples['left'][3])
            finally:
                bridge.close()

    def test_forearm_guides_elbow_without_displacing_wrist(self):
        m,d,c,_=setup('human');s='right'
        before=d.body(f'{s}_elbow').xpos.copy()
        other=d.body('left_elbow').xpos.copy()
        axis=c.home[s]-d.body(f'{s}_shoulder').xpos
        axis/=np.linalg.norm(axis)
        target=rotation_matrix(axis*.3)@c.forearm_direction(d,s)
        c.set_forearm_target(s,target)
        for _ in range(650):
            c.command(d,active=(s,));advance_physics(m,d,c)
        actual=c.forearm_direction(d,s)
        self.assertLess(np.arccos(np.clip(actual@target,-1,1)),.03)
        self.assertGreater(np.linalg.norm(d.body(f'{s}_elbow').xpos-before),.025)
        self.assertLess(np.linalg.norm(d.site_xpos[c.sites[s]]-c.home[s]),.003)
        np.testing.assert_allclose(d.body('left_elbow').xpos,other,atol=.002)
        c.set_forearm_target(s,None)
        for _ in range(30):
            c.command(d);advance_physics(m,d,c)
        self.assertIsNone(c.forearm_directions[s])
        self.assertTrue(np.isfinite(d.qpos).all())
        c.reset(d)
        self.assertTrue(all(v is None for v in c.forearm_targets.values()))

    def test_virtual_wrist_can_physically_roll_180_and_cross_wrap(self):
        m,d,c,_=setup('human')
        for s in ('left','right'):
            c.set_rotation(s,rotation_matrix([np.pi,0,0])@c.home_rotations[s])
            self.assertAlmostEqual(np.linalg.norm(rotation_vector(c.rotation_targets[s]@c.home_rotations[s].T)),np.pi)
        for _ in range(750):
            c.command(d);advance_physics(m,d,c)
        for s in ('left','right'):
            actual=d.site_xmat[c.sites[s]].reshape(3,3)
            self.assertLess(np.linalg.norm(rotation_vector(c.rotation_targets[s]@actual.T)),.02)
            self.assertLess(np.linalg.norm(d.site_xpos[c.sites[s]]-c.home[s]),.003)
        # Crossing +/-180 must not create a full-turn jump in the command.
        previous=c.rotations['right'].copy()
        c.set_rotation('right',rotation_matrix([np.deg2rad(181),0,0])@c.home_rotations['right'])
        c.command(d)
        self.assertLessEqual(np.linalg.norm(rotation_vector(c.rotations['right']@previous.T)),.0250001)
        self.assertEqual(d.ncon,0)


if __name__=='__main__':
    unittest.main()
