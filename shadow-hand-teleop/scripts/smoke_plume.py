"""Lightweight buoyant smoke particles with MuJoCo surface collisions.

A visual flow approximation, not a volumetric fluid solver. Smoke does not
apply forces to the prescribed replay motion.
"""
import mujoco
import numpy as np


class SmokePlume:
    def __init__(self, model, emitters, count=96):
        self.model = model
        self.emitters = np.asarray(emitters, dtype=float)
        self.rng = np.random.default_rng(7)
        self.positions = np.zeros((count, 3))
        self.velocities = np.zeros((count, 3))
        self.ages = np.full(count, 99.)
        self.last_time = None
        self.carry = 0.
        self.cursor = 0
        self.collisions = 0
        self.probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'smoke_probe_geom')
        if self.probe < 0:
            raise ValueError('Smoke requires the hidden smoke_probe_geom distance probe')
        self.query_data = mujoco.MjData(model)
        self.obstacles = np.array([i for i in range(model.ngeom)
            if i != self.probe and (model.geom_contype[i] or model.geom_conaffinity[i]
                                    or model.geom_group[i] == 2)], dtype=int)
        self.segment = np.zeros(6)

    @staticmethod
    def radius(age):
        return .009 + .005 * age

    def project_particles(self, indices):
        """Batch clearance against conservative oriented bounds of hand geoms.

        Bounds move with the replay and enclose the visible meshes. This avoids
        expensive per-particle mesh queries while preserving solid exclusion.
        """
        if not len(indices) or not len(self.obstacles):
            return
        ids = self.obstacles
        rotations = self.query_data.geom_xmat[ids].reshape(-1, 3, 3)
        centers = self.query_data.geom_xpos[ids]
        local_centers = self.model.geom_aabb[ids, :3]
        half = self.model.geom_aabb[ids, 3:]
        planes = self.model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_PLANE
        radii = self.radius(self.ages[indices])
        for iteration in range(5):
            offsets = self.positions[indices, None, :] - centers[None, :, :]
            local = np.einsum('ngi,gij->ngj', offsets, rotations) - local_centers
            q = np.abs(local) - half
            outside = np.maximum(q, 0)
            length = np.linalg.norm(outside, axis=2)
            distance = length + np.minimum(np.max(q, axis=2), 0)
            normals = np.sign(local) * outside / np.maximum(length[..., None], 1e-12)
            inside = length < 1e-10
            axis = np.argmax(q, axis=2)
            rows, cols = np.where(inside)
            normals[rows, cols] = 0
            normals[rows, cols, axis[rows, cols]] = np.where(local[rows, cols, axis[rows, cols]] >= 0, 1., -1.)
            if np.any(planes):
                distance[:, planes] = local[:, planes, 2]
                normals[:, planes] = [0., 0., 1.]
            nearest = np.argmin(distance, axis=1)
            row = np.arange(len(indices))
            clearance = distance[row, nearest]
            hit = clearance < radii + .002
            if not np.any(hit):
                return
            selected = indices[hit]
            local_normal = normals[row[hit], nearest[hit]]
            normal = np.einsum('nij,nj->ni', rotations[nearest[hit]], local_normal)
            self.positions[selected] += normal * (radii[hit] + .003 - clearance[hit])[:, None]
            velocity = self.velocities[selected]
            inward = np.minimum(np.sum(velocity * normal, axis=1), 0)
            tangent = self.rng.normal(0, .045, velocity.shape)
            tangent -= np.sum(tangent * normal, axis=1)[:, None] * normal
            self.velocities[selected] = velocity - inward[:, None] * normal + tangent + normal * .025
            self.collisions += len(selected)
        # Suppress puffs trapped between overlapping bounds.
        self.ages[selected] = 99.

    def advance(self, data, stamp):
        if self.last_time is None or stamp < self.last_time:
            self.last_time = stamp
            self.ages[:] = 99.
            return
        elapsed = stamp - self.last_time
        self.last_time = stamp
        # Bound per-frame work if a recording contains a long tracking gap.
        elapsed = min(elapsed, .25)
        while elapsed > 1e-9:
            dt = min(elapsed, 1 / 60)
            self.step(data, dt)
            elapsed -= dt

    def step(self, data, dt):
        self.query_data.geom_xpos[:] = data.geom_xpos
        self.query_data.geom_xmat[:] = data.geom_xmat
        self.carry += dt * 24
        while self.carry >= 1:
            self.carry -= 1
            i = self.cursor % len(self.ages)
            self.cursor += 1
            emitter = self.emitters[self.cursor % len(self.emitters)]
            self.positions[i] = emitter + self.rng.normal(0, [.055, .055, .006])
            self.velocities[i] = self.rng.normal(0, [.025, .025, .015]) + [0, 0, .22]
            self.ages[i] = 0
        active = np.flatnonzero(self.ages < 4.)
        self.ages[active] += dt
        if not len(active):
            return
        points = self.positions[active]
        ages = self.ages[active]
        flow = np.column_stack((.05 * np.sin(points[:, 2] * 9 + ages * 2),
                                .05 * np.cos(points[:, 0] * 8 - ages * 2),
                                np.full(len(active), .28)))
        self.velocities[active] += (flow - self.velocities[active]) * min(1., dt * 2)
        self.project_particles(active)
        max_delta = np.max(np.linalg.norm(self.velocities[active], axis=1)) * dt
        steps = max(1, int(np.ceil(max_delta / (.5 * np.min(self.radius(ages))))))
        for _ in range(steps):
            active = active[self.ages[active] < 4.]
            if not len(active):
                break
            self.positions[active] += self.velocities[active] * (dt / steps)
            self.project_particles(active)

    def draw(self, scene):
        for i in np.flatnonzero(self.ages < 4.):
            if scene.ngeom >= scene.maxgeom:
                break
            age = self.ages[i]
            alpha = .19 * min(1., age / .15) * min(1., (4. - age) / .8)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, self.radius(age)), self.positions[i],
                np.eye(3).flatten(), np.array([.72, .79, .87, alpha]))
            scene.ngeom += 1
