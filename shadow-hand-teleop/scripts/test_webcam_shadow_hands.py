import unittest

import numpy as np

from hand_retargeting import finger_targets, synthetic_skeleton
from webcam_shadow_hands import mediapipe_skeleton


def mediapipe_points(angle=0.0):
    """21 MediaPipe-ordered landmarks bent like synthetic_skeleton, without metacarpals."""
    joints = synthetic_skeleton(angle)[:, :3, 3]
    points = np.zeros((21, 3))
    points[0:5] = joints[0:5]
    for finger in range(4):
        start = 5 + 5 * finger
        # Wrist-to-knuckle must line up with the old metacarpal bone.
        points[5 + 4 * finger : 9 + 4 * finger] = joints[start + 1 : start + 5]
    points[0] = joints[5] - (joints[6] - joints[5])
    return points


class WebcamMappingTests(unittest.TestCase):
    def test_layout(self):
        points = np.arange(63, dtype=float).reshape(21, 3)
        joints = mediapipe_skeleton(points)
        self.assertEqual(joints.shape, (25, 4, 4))
        np.testing.assert_array_equal(joints[1:5, :3, 3], points[1:5])
        np.testing.assert_array_equal(joints[10, :3, 3], points[0])
        np.testing.assert_array_equal(joints[11:15, :3, 3], points[9:13])

    def test_straight_and_bent(self):
        straight = finger_targets(mediapipe_skeleton(mediapipe_points()))
        bent = finger_targets(mediapipe_skeleton(mediapipe_points(0.6)))
        self.assertAlmostEqual(straight["FFJ0"], 0)
        self.assertAlmostEqual(bent["FFJ3"], 0.6)
        self.assertAlmostEqual(bent["FFJ0"], 1.2)
        self.assertGreater(bent["THJ1"], straight["THJ1"])

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            mediapipe_skeleton(np.zeros((25, 3)))


if __name__ == "__main__":
    unittest.main()
