"""Two gravity-compensated Panda arms carrying the existing Shadow Hands.

Uses the official Menagerie models (downloaded to the ignored local cache).
The Cartesian controller shares orientation between two Shadow wrist hinges
and seven Panda joints. Finger targets continue to use hand_retargeting.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import mujoco_menagerie as menagerie
import numpy as np

from view_dual_shadow_hands import BASE_SCENE


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MENAGERIE_CACHE_DIR", str(ROOT / ".cache/mujoco_menagerie"))
SIDES = ("left", "right")
HOME = np.array([0.0, -0.55, 0.0, -1.8, 0.0, 1.25, 0.785398])
# Shadow fingers extend along local +Z and flex toward local -Y.
# Map fingertips forward (+X), palms down (-Z), mirrored thumbs inward.
PALM_DOWN_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
# avp_stream already converts Apple Y-up into Z-up. Here +Y (forward)
# becomes robot +X, and +X (user's right) becomes robot -Y. Face forward
# in the tracking-origin frame when calibrating; C resets positional offsets.
TRACKING_TO_ROBOT = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


def wrist_transform(transform) -> np.ndarray:
    """Validate either SDK wrist shape, without accepting an all-zero pose."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite wrist transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4):
        raise ValueError("Invalid homogeneous wrist transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.05) or np.linalg.det(rotation) < 0.9:
        raise ValueError("Invalid wrist rotation")
    return matrix.copy()


def wrist_position(transform) -> np.ndarray:
    return wrist_transform(transform)[:3, 3]


def wrist_rotation(transform) -> np.ndarray:
    rotation = wrist_transform(transform)[:3, :3]
    # Remove small SDK numerical drift after checking the original pose.
    u, _, vt = np.linalg.svd(rotation)
    return u @ vt


def rotation_vector(rotation):
    """SO(3) logarithm, including near-180-degree rotations."""
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation).ravel())
    if quaternion[0] < 0:
        quaternion *= -1
    length = np.linalg.norm(quaternion[1:])
    if length < 1e-10:
        return np.zeros(3)
    return quaternion[1:] * (2 * np.arctan2(length, quaternion[0]) / length)


def rotation_matrix(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-10:
        return np.eye(3)
    quaternion = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quaternion, vector / angle, angle)
    result = np.zeros(9)
    mujoco.mju_quat2Mat(result, quaternion)
    return result.reshape(3, 3)


def build_arm_model() -> mujoco.MjModel:
    panda_path = menagerie.get("franka_emika_panda").path()
    hand_path = menagerie.get("shadow_hand").path()
    scene = mujoco.MjSpec.from_string(BASE_SCENE)
    scene.modelname = "dual_panda_shadow_hands"
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.worldbody.add_geom(
        name="worktop", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.62, 0, 0.48], size=[0.46, 0.58, 0.025],
        rgba=[0.55, 0.39, 0.26, 1], friction=[0.8, 0.005, 0.0001])
    for lateral in (-0.48, 0.48):
        for forward in (0.23, 1.01):
            scene.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, pos=[forward, lateral, 0.18],
                size=[0.025, 0.025, 0.275], rgba=[0.18, 0.19, 0.21, 1])
    for side, lateral in (("left", 0.25), ("right", -0.25)):
        arm = mujoco.MjSpec.from_file(str(panda_path / "panda_nohand.xml"))
        # Initialize explicitly below instead of merging differently sized keys.
        for key in list(arm.keys):
            arm.delete(key)
        reference_model = arm.compile()
        reference = mujoco.MjData(reference_model)
        reference.qpos[:] = HOME
        mujoco.mj_forward(reference_model, reference)
        hand = mujoco.MjSpec.from_file(str(hand_path / f"{side}_hand.xml"))
        forearm = hand.body(f"{side[0]}h_forearm")
        wrist = hand.body(f"{side[0]}h_wrist")
        # Panda supplies the forearm. Replace the redundant Shadow housing
        # with a small adapter, retaining the original wrist/finger joints.
        for geom in list(forearm.geoms):
            hand.delete(geom)
        forearm.quat = [1, 0, 0, 0]
        forearm.mass = 0.02
        forearm.ipos = [0, 0, 0]
        forearm.inertia = [1e-5, 1e-5, 1e-5]
        forearm.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         pos=[0, 0, -0.03], size=[0.022, 0.03, 0], rgba=[0.12, 0.13, 0.15, 1],
                         contype=0, conaffinity=0)
        wrist.pos = [0, 0, 0]
        native_wrist = np.zeros(9)
        mujoco.mju_quat2Mat(native_wrist, wrist.quat / np.linalg.norm(wrist.quat))
        flange = reference.body("attachment").xmat.reshape(3, 3)
        mount_rotation = flange.T @ PALM_DOWN_ROTATION @ native_wrist.reshape(3, 3).T
        mount_quat = np.zeros(4)
        mujoco.mju_mat2Quat(mount_quat, mount_rotation.ravel())
        base = scene.worldbody.add_frame(name=f"{side}_base", pos=[0.0, lateral, 0])
        scene.attach(arm, prefix=f"{side}_arm_", frame=base)
        mount = scene.body(f"{side}_arm_attachment").add_frame(
            name=f"{side}_hand_mount", quat=mount_quat,
            pos=flange.T @ np.array([0.06, 0, 0]))
        hand.body(f"{side[0]}h_palm").add_site(
            name="wrist_site", size=[0.006, 0, 0], rgba=[0.1, 0.8, 0.9, 1])
        scene.attach(hand, prefix=f"{side}_", frame=mount)
        # The touching mounting surfaces are a bolted assembly, not an impact.
        for link in ("link7", "attachment"):
            for part in ("forearm", "wrist"):
                scene.add_exclude(bodyname1=f"{side}_arm_{link}",
                                  bodyname2=f"{side}_{side[0]}h_{part}")
        goal = scene.worldbody.add_body(name=f"{side}_goal", mocap=True)
        goal.add_geom(name=f"{side}_goal_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[0.015, 0, 0], rgba=[0.15, 0.9, 0.55, 0.55],
                      contype=0, conaffinity=0)
        for axis, color in zip(np.eye(3), ([1, .2, .2, .6], [.2, 1, .2, .6], [.2, .4, 1, .6])):
            goal.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.002, 0, 0],
                          fromto=np.concatenate([np.zeros(3), 0.055 * axis]),
                          rgba=color, contype=0, conaffinity=0)
    # Idealized gravity compensation for this simulated assembly.
    for body in scene.bodies:
        if not body.mocap:
            body.gravcomp = 1
    return scene.compile()


class ArmController:
    """Bounded damped-least-squares IK plus joint-position actuator targets."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.joints = {}
        self.qids = {}
        self.dofs = {}
        self.actuators = {}
        self.sites = {}
        self.goals = {}
        self.rotations = {}
        self.rotation_targets = {}
        self.wrist_qids = {}
        self.wrist_acts = {}
        self.wrist_limits = {}
        self.targets = {}
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        for side in SIDES:
            joints = np.array([model.joint(f"{side}_arm_joint{i}").id for i in range(1, 8)])
            self.joints[side] = joints
            self.qids[side] = model.jnt_qposadr[joints]
            self.dofs[side] = model.jnt_dofadr[joints]
            self.actuators[side] = np.array([
                model.actuator(f"{side}_arm_actuator{i}").id for i in range(1, 8)])
            self.sites[side] = model.site(f"{side}_wrist_site").id
            wrist_joints = [model.joint(f"{side}_{side[0]}h_{name}").id for name in ("WRJ1", "WRJ2")]
            self.wrist_qids[side] = model.jnt_qposadr[wrist_joints]
            self.wrist_acts[side] = np.array([
                model.actuator(f"{side}_{side[0]}h_A_{name}").id for name in ("WRJ1", "WRJ2")])
            self.wrist_limits[side] = model.actuator_ctrlrange[self.wrist_acts[side]]
            self.goals[side] = int(model.body_mocapid[model.body(f"{side}_goal").id])
        self.reset(data)

    def reset(self, data):
        mujoco.mj_resetData(self.model, data)
        for side in SIDES:
            data.qpos[self.qids[side]] = HOME
            data.ctrl[self.actuators[side]] = HOME
        mujoco.mj_forward(self.model, data)
        self.home = {}
        self.home_rotations = {}
        for side in SIDES:
            site = self.sites[side]
            self.home[side] = data.site_xpos[site].copy()
            self.targets[side] = self.home[side].copy()
            self.rotations[side] = data.site_xmat[site].reshape(3, 3).copy()
            self.home_rotations[side] = self.rotations[side].copy()
            self.rotation_targets[side] = self.rotations[side].copy()
        self.update_markers(data)

    def set_target(self, side, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Target must be a finite 3D position")
        # Local demonstration volume; keep the two hands on their own sides.
        offset = np.clip(position - self.home[side], [-0.16, -0.14, -0.16], [0.16, 0.14, 0.18])
        self.targets[side] = self.home[side] + offset

    def set_rotation(self, side, rotation):
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError("Orientation must be a proper rotation matrix")
        # Older hardware-shaped prototypes retain their 120-degree limit.
        limit = getattr(self, "orientation_limit", np.deg2rad(120))
        vector = rotation_vector(rotation @ self.home_rotations[side].T)
        angle = np.linalg.norm(vector)
        if angle > limit:
            vector *= limit / angle
        self.rotation_targets[side] = rotation_matrix(vector) @ self.home_rotations[side]

    def wrist_angles(self, side, rotation):
        # Local Z is the forearm/finger direction. Z-Y-X decomposition puts
        # forearm roll into the Panda and the remaining tilts into the two
        # Shadow hinges. Arm IK handles any residual after wrist clipping.
        local = self.home_rotations[side].T @ rotation
        bend = np.arctan2(local[2, 1], local[2, 2])
        deviation = -np.arcsin(np.clip(-local[2, 0], -1, 1))
        limits = self.wrist_limits[side]
        return np.clip([bend, deviation], limits[:, 0], limits[:, 1])

    def update_markers(self, data):
        for side in SIDES:
            data.mocap_pos[self.goals[side]] = self.targets[side]
            mujoco.mju_mat2Quat(data.mocap_quat[self.goals[side]], self.rotations[side].ravel())

    def solve(self, data, side, dt=0.01):
        """Return an IK target limited to 0.6 rad/s from the last command."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        qids, dofs = self.qids[side], self.dofs[side]
        acts, site = self.actuators[side], self.sites[side]
        limits = self.model.jnt_range[self.joints[side]]
        self.scratch.qpos[:] = data.qpos
        # Warm-start from the last command rather than integrate IK faster
        # than physics can follow when the target is outside the workspace.
        self.scratch.qpos[qids] = data.ctrl[acts]
        self.scratch.qpos[self.wrist_qids[side]] = data.ctrl[self.wrist_acts[side]]
        for _ in range(12):
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
            rotation = self.scratch.site_xmat[site].reshape(3, 3)
            angle = rotation_vector(self.rotations[side] @ rotation.T)
            error = np.concatenate([self.targets[side] - self.scratch.site_xpos[site], 0.3 * angle])
            mujoco.mj_jacSite(self.model, self.scratch, self.jacp, self.jacr, site)
            jacobian = np.vstack([self.jacp[:, dofs], 0.3 * self.jacr[:, dofs]])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.003**2 * np.eye(6), error)
            self.scratch.qpos[qids] = np.clip(
                self.scratch.qpos[qids] + np.clip(delta, -0.04, 0.04),
                limits[:, 0] + 0.01, limits[:, 1] - 0.01)
        command = data.ctrl[acts] + np.clip(self.scratch.qpos[qids] - data.ctrl[acts], -0.6 * dt, 0.6 * dt)
        return np.clip(command, limits[:, 0] + 0.01, limits[:, 1] - 0.01)

    def command(self, data, active=SIDES, dt=0.01):
        for side in active:
            delta = rotation_vector(self.rotation_targets[side] @ self.rotations[side].T)
            angle = np.linalg.norm(delta)
            if angle > 0.8 * dt:
                delta *= 0.8 * dt / angle
            self.rotations[side] = rotation_matrix(delta) @ self.rotations[side]
            acts = self.wrist_acts[side]
            desired = self.wrist_angles(side, self.rotations[side])
            data.ctrl[acts] += np.clip(desired - data.ctrl[acts], -0.6 * dt, 0.6 * dt)
            data.ctrl[self.actuators[side]] = self.solve(data, side, dt)
        self.update_markers(data)


class WristMapper:
    """Relative wrist position/orientation, with explicit calibration.

    A stale stream invalidates the calibration for that hand. Fresh tracking
    alone cannot move the arm again; the operator must press C to recenter.
    (The viewer's optional --auto-recenter calls recenter() after short gaps.)
    """

    def __init__(self, scale=0.6):
        if not np.isfinite(scale) or not 0 < scale <= 2:
            raise ValueError("Wrist scale must be in (0, 2]")
        self.scale = scale
        self.origins = {}
        self.anchors = {}
        self.rotation_origins = {}
        self.rotation_anchors = {}
        self.forearm_alignments = {}
        self.position_scales = {}
        # Gains a side had when it was last invalidated, for recenter().
        self.previous_scales = {}

    def calibrate(self, side, wrist, robot_position, robot_rotation=None,
                  forearm=None, robot_forearm=None):
        self.origins[side] = wrist_position(wrist)
        self.position_scales[side] = np.full(3,self.scale,dtype=float)
        self.anchors[side] = np.asarray(robot_position, dtype=float).copy()
        self.rotation_origins[side] = wrist_rotation(wrist)
        if robot_rotation is not None:
            self.rotation_anchors[side] = np.asarray(robot_rotation, dtype=float).copy()
        else:
            self.rotation_anchors.pop(side, None)
        self.forearm_alignments.pop(side, None)
        if forearm is not None and robot_forearm is not None:
            from forearm_tracking import unit_vector
            source = unit_vector(TRACKING_TO_ROBOT @ unit_vector(forearm))
            target = unit_vector(robot_forearm)
            cross = np.cross(source, target)
            cosine = np.clip(np.dot(source, target), -1, 1)
            if np.linalg.norm(cross) > 1e-8:
                axis = cross / np.linalg.norm(cross)
                alignment = rotation_matrix(axis * np.arccos(cosine))
            elif cosine > 0:
                alignment = np.eye(3)
            else:
                axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
                alignment = rotation_matrix(unit_vector(axis) * np.pi)
            self.forearm_alignments[side] = alignment

    def calibrate_hand_span(self):
        """Match lateral closure after BOTH hands were calibrated together.

        Independent initial origins alone preserve an artificial robot gap.
        Use per-side gains so losing one hand never jumps the other hand.
        This is a lateral span fit, not a body-heading or anatomy estimate.
        """
        if not all(side in self.origins for side in SIDES):
            return False
        human=(TRACKING_TO_ROBOT @ (self.origins['left']-self.origins['right']))[1]
        robot=(self.anchors['left']-self.anchors['right'])[1]
        if not .12<=human<=.85 or robot<.08:
            return False
        gain=robot/human
        if not .5<=gain<=3.:
            return False
        for side in SIDES:
            self.position_scales[side][1]=gain
        return True

    def recenter(self, side, wrist, robot_position, robot_rotation=None,
                 forearm=None, robot_forearm=None):
        """Re-anchor a side lost after calibration, keeping its previous gains.

        Same as calibrate() at the robot's current pose, except that the
        per-axis position scales (including the hand-span lateral gain) are
        restored instead of reset to `scale`. Returns False, changing
        nothing, if the side was not calibrated when it was last invalidated.
        """
        scales = self.previous_scales.get(side)
        if scales is None:
            return False
        self.calibrate(side, wrist, robot_position, robot_rotation,
                       forearm=forearm, robot_forearm=robot_forearm)
        self.position_scales[side] = scales.copy()
        return True

    def invalidate(self, side):
        if side in self.position_scales:
            self.previous_scales[side] = self.position_scales[side].copy()
        self.origins.pop(side, None)
        self.anchors.pop(side, None)
        self.rotation_origins.pop(side, None)
        self.rotation_anchors.pop(side, None)
        self.forearm_alignments.pop(side, None)
        self.position_scales.pop(side, None)

    def forearm_target(self, side, forearm):
        if forearm is None or side not in self.forearm_alignments:
            return None
        from forearm_tracking import unit_vector
        return self.forearm_alignments[side] @ TRACKING_TO_ROBOT @ unit_vector(forearm)

    def target(self, side, wrist):
        if side not in self.origins:
            return None
        delta = wrist_position(wrist) - self.origins[side]
        return self.anchors[side] + self.position_scales[side] * (TRACKING_TO_ROBOT @ delta)

    def rotation_target(self, side, wrist):
        if side not in self.rotation_anchors:
            return None
        delta = wrist_rotation(wrist) @ self.rotation_origins[side].T
        world_delta = TRACKING_TO_ROBOT @ delta @ TRACKING_TO_ROBOT.T
        return world_delta @ self.rotation_anchors[side]
