"""World-space tracked skeletons beside wrist-following Shadow Hands."""
import mujoco
import numpy as np

# ARKit Y-up -> MuJoCo Z-up, preserving handedness and metric scale.
WORLD = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
EDGES = [(0, 1)] + [(i, i + 1) for i in (1, 2, 3)]
for start in (5, 10, 15, 20):
    EDGES += [(0, start)] + [(i, i + 1) for i in range(start, start + 4)]


def basis(wrist, middle, index, little):
    forward = middle - wrist
    forward /= np.linalg.norm(forward)
    across = index - little
    across -= forward * np.dot(across, forward)
    across /= np.linalg.norm(across)
    return np.column_stack((across, np.cross(forward, across), forward))


def quaternion_matrix(xyzw):
    quat = np.asarray(xyzw)[[3, 0, 1, 2]].copy()
    norm = np.linalg.norm(quat)
    if not np.isfinite(quat).all() or norm < 1e-8:
        raise ValueError('Invalid wrist quaternion')
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat / norm)
    return matrix.reshape(3, 3)


class ReplayComparison:
    def __init__(self, arrays, model):
        self.arrays = arrays
        self.smoke = True
        points = np.concatenate([arrays[f'hand_{side}'].reshape(-1, 3)
                                 for side in ('left', 'right') if f'hand_{side}' in arrays])
        points = points[np.isfinite(points).all(axis=1)] @ WORLD.T
        self.center = (points.min(axis=0) + points.max(axis=0)) / 2
        self.separation = max(0.7, np.ptp(points[:, 0]) + 0.45)
        self.lookat = np.array([0., 0., 0.65])
        self.distance = max(1.6, self.separation * 2.1, np.max(np.ptp(points, axis=0)) * 2.2)
        self.calibration = {}
        self.palm = {}
        self.mocap = {}
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        for side in ('left', 'right'):
            prefix = f'{side}_{side[0]}h_'
            self.mocap[side] = model.body(f'{side}_mount').mocapid[0]
            palm = reference.xpos[model.body(prefix + 'palm').id].copy()
            self.palm[side] = palm
            robot_basis = basis(palm,
                reference.xpos[model.body(prefix + 'mfknuckle').id],
                reference.xpos[model.body(prefix + 'ffknuckle').id],
                reference.xpos[model.body(prefix + 'lfmetacarpal').id])
            if f'hand_{side}' not in arrays:
                continue
            for positions, quat in zip(arrays[f'hand_{side}'], arrays[f'wrist_{side}']):
                if not np.isfinite(positions).all():
                    continue
                try:
                    recorded_basis = basis(positions[0], positions[10], positions[5], positions[20])
                    wrist_rotation = quaternion_matrix(quat)
                except ValueError:
                    continue
                if not np.isfinite(recorded_basis).all():
                    continue
                self.calibration[side] = wrist_rotation.T @ recorded_basis @ robot_basis.T
                break

        from smoke_plume import SmokePlume
        emitters = []
        for side in self.calibration:
            wrists = arrays[f'hand_{side}'][:, 0]
            wrists = wrists[np.isfinite(wrists).all(axis=1)]
            source = self.position(np.median(wrists, axis=0), 1)
            source[2] = max(.03, self.position(wrists[np.argmin(wrists[:, 1])], 1)[2] - .25)
            emitters.append(source)
        if emitters:
            self.plume = SmokePlume(model, emitters)

    def advance_smoke(self, data, stamp):
        if hasattr(self, 'plume'):
            if self.smoke:
                self.plume.advance(data, stamp)
            else:
                self.plume.last_time = stamp
                self.plume.ages[:] = 99.

    def position(self, point, panel):
        return WORLD @ point - self.center + self.lookat + np.array([panel * self.separation / 2, 0., 0.])

    def update(self, data, frame):
        for side, mocap in self.mocap.items():
            positions = self.arrays.get(f'hand_{side}')
            if side not in self.calibration or positions is None:
                data.mocap_pos[mocap] = [0, 0, -10]
                continue
            if not np.isfinite(positions[frame, 0]).all():
                # Hold the last wrist pose across missing tracking.
                continue
            try:
                rotation = WORLD @ quaternion_matrix(self.arrays[f'wrist_{side}'][frame]) @ self.calibration[side]
            except ValueError:
                continue
            data.mocap_pos[mocap] = self.position(positions[frame, 0], 1) - rotation @ self.palm[side]
            mujoco.mju_mat2Quat(data.mocap_quat[mocap], rotation.flatten())

    def draw(self, scene, frame):
        scene.ngeom = 0
        for side, color in (('left', [0.2, 0.8, 1., 1.]), ('right', [1., 0.55, 0.15, 1.])):
            positions = self.arrays.get(f'hand_{side}')
            if positions is None:
                continue
            points = np.array([self.position(p, -1) for p in positions[frame]])
            for a, b in EDGES:
                if not np.isfinite(points[[a, b]]).all():
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(color))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, points[a], points[b])
                scene.ngeom += 1
        if self.smoke and hasattr(self, "plume"):
            self.plume.draw(scene)
        for panel, label in ((-1, 'Recorded tracking'), (1, 'Shadow reconstruction')):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                self.lookat + [panel * self.separation / 2, 0., .5], np.eye(3).flatten(), np.ones(4))
            geom.label = label
            scene.ngeom += 1
