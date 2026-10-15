import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from replay_data import load_demo, save_demo, validate_model


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.arrays = dict(timestamp=np.array([0., .02]), qpos=np.zeros((2, 3)),
                           qvel=np.ones((2, 2)), action=np.ones((2, 1)))

    def test_round_trip_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.npz', '.h5', '.hdf5'):
                path = Path(directory) / ('demo' + suffix)
                save_demo(path, self.arrays, {'source': 'test'})
                loaded = load_demo(path)
                for name, values in self.arrays.items():
                    np.testing.assert_array_equal(loaded[name], values)
                if suffix != '.npz':
                    with h5py.File(path) as file:
                        self.assertEqual(file.attrs['schema_version'], 1)
                        self.assertIn('test', file.attrs['metadata'])
                validate_model(loaded, SimpleNamespace(nq=3, nv=2, nu=1))
                with self.assertRaisesRegex(ValueError, 'qpos'):
                    validate_model(loaded, SimpleNamespace(nq=4, nv=2, nu=1))

    def test_invalid_files(self):
        invalid = [dict(timestamp=np.array([0., 0.])),
                   dict(timestamp=np.array([0., np.nan])),
                   dict(qpos=np.zeros((1, 3))), dict(action=np.array([1., 2.]))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.h5'
            for overrides in invalid:
                save_demo(path, self.arrays | overrides, {})
                with self.assertRaises(ValueError):
                    load_demo(path)
            with h5py.File(path, 'w') as file:
                file.create_dataset('timestamp', data=[0.])
            with self.assertRaisesRegex(ValueError, 'Missing'):
                load_demo(path)

    def test_holos_joint_order_and_missing_tracking(self):
        from holos_replay import THUMB, FINGERS, retarget_holos
        from hand_retargeting import synthetic_skeleton
        from view_dual_shadow_hands import build_model
        names = THUMB + FINGERS + ['forearmWrist', 'forearmArm']
        skeleton = synthetic_skeleton(0.5)[:, :3, 3]
        poses = np.zeros((3, 26, 7))
        poses[:, :24, :3] = skeleton[1:]
        poses[:, 24, :3] = skeleton[0]
        poses[0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'holos.h5'
            with h5py.File(path, 'w') as file:
                file.attrs['holos_convention_version'] = 'HolosCapture-1.0'
                file.create_dataset('joint', data=names)
                file.create_dataset('data/demo_0/time', data=[0., .02, .04])
                file.create_dataset('data/demo_0/obs/hand_right_joints_world', data=poses)
            loaded = load_demo(path)
            np.testing.assert_array_equal(loaded['hand_right'][1], skeleton)
            model = build_model()
            replay = retarget_holos(loaded, model)
            validate_model(replay, model)
            self.assertTrue(np.isfinite(replay['qpos']).all())
            self.assertTrue(np.any(replay['action'][-1] != 0))
            np.testing.assert_array_equal(replay['action'][0], np.zeros(model.nu))
            with self.assertRaisesRegex(ValueError, 'Missing Holos demonstration'):
                load_demo(path, 'demo_99')

    def test_comparison_tracks_translation_rotation_and_missing_frames(self):
        import mujoco
        from replay_comparison import ReplayComparison, WORLD
        from view_dual_shadow_hands import build_model
        points = np.zeros((25, 3))
        points[5] = [-.03, .08, 0]
        points[10] = [0, .09, 0]
        points[20] = [.03, .07, 0]
        rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
        arrays = {'hand_right': np.array([points, points @ rotation.T + [1, 2, 3], points * np.nan]),
                  'wrist_right': np.array([[0, 0, 0, 1], [0, 0, 2**-.5, 2**-.5], [np.nan]*4])}
        model = build_model(moving=True)
        data = mujoco.MjData(model)
        view = ReplayComparison(arrays, model)
        view.update(data, 0)
        first_quat = data.mocap_quat[view.mocap['right']].copy()
        view.update(data, 1)
        mujoco.mj_forward(model, data)
        palm = data.xpos[model.body('right_rh_palm').id]
        np.testing.assert_allclose(palm, view.position(arrays['hand_right'][1, 0], 1), atol=1e-10)
        first_matrix = np.empty(9)
        mujoco.mju_quat2Mat(first_matrix, first_quat)
        actual = np.empty(9)
        mujoco.mju_quat2Mat(actual, data.mocap_quat[view.mocap['right']])
        np.testing.assert_allclose(actual.reshape(3, 3), WORLD @ rotation @ WORLD.T @ first_matrix.reshape(3, 3), atol=1e-10)
        held = data.mocap_pos.copy()
        view.update(data, 2)
        np.testing.assert_array_equal(data.mocap_pos, held)
        scene = mujoco.MjvScene(model, maxgeom=100)
        view.draw(scene, 2)
        self.assertEqual(scene.ngeom, 2)  # Labels only: missing tracking is hidden.


if __name__ == '__main__':
    unittest.main()
