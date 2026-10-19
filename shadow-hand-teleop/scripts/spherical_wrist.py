"""Simplified forearms with actual MuJoCo ball joints and Shadow fingers.

This modifies the Panda assembly: joints 6/7 and their meshes are removed,
as are Shadow's two original wrist hinges. Three torque motors actuate each
ball joint; the five remaining arm joints position its center. The geometry
is an interaction prototype, not a hardware-faithful Panda or human wrist.
"""

import mujoco
import mujoco_menagerie as menagerie
import numpy as np

from arm_control import (ArmController, BASE_SCENE, HOME, PALM_DOWN_ROTATION,
                         SIDES, rotation_matrix, rotation_vector)


def quaternion(rotation):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(rotation).ravel())
    return result


def build_spherical_model():
    panda_path = menagerie.get("franka_emika_panda").path()
    hand_path = menagerie.get("shadow_hand").path()
    scene = mujoco.MjSpec.from_string(BASE_SCENE)
    scene.modelname = "dual_spherical_wrist_shadow_hands"
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.worldbody.add_geom(name="worktop", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[.62, 0, .48], size=[.46, .58, .025], rgba=[.55, .39, .26, 1],
        friction=[.8, .005, .0001])
    for lateral in (-.48, .48):
        for forward in (.23, 1.01):
            scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[forward, lateral, .18], size=[.025, .025, .275], rgba=[.18, .19, .21, 1])
    for side, lateral in (("left", .25), ("right", -.25)):
        arm = mujoco.MjSpec.from_file(str(panda_path / "panda_nohand.xml"))
        for key in list(arm.keys):
            arm.delete(key)
        for i in (6, 7):
            arm.delete(arm.actuator(f"actuator{i}"))
        arm.delete(arm.body("link6"))
        reference_model = arm.compile()
        reference = mujoco.MjData(reference_model)
        reference.qpos[:] = HOME[:5]
        mujoco.mj_forward(reference_model, reference)
        # A slim capsule replaces the industrial wrist motor housings.
        forearm = arm.body("link5").add_body(name="simple_forearm",
            quat=quaternion(reference.body("link5").xmat.reshape(3, 3).T))
        pivot = np.array([.14, 0, -.08])
        forearm.add_geom(name="forearm_sleeve", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.concatenate([np.zeros(3), pivot]), size=[.025, 0, 0],
            rgba=[.65, .68, .72, 1], density=450)
        ball = forearm.add_body(name="spherical_wrist", pos=pivot,
                               quat=quaternion(PALM_DOWN_ROTATION))
        ball.add_joint(name="wrist_ball", type=mujoco.mjtJoint.mjJNT_BALL,
                       limited=1, range=[0, np.deg2rad(130)],
                       armature=.003, damping=.08)
        ball.add_geom(name="wrist_ball_surface", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[.028, 0, 0], rgba=[.2, .24, .29, 1], density=900)
        ball.add_site(name="wrist_site", size=[.004, 0, 0], rgba=[.1, .8, .9, 1])
        for axis, name in zip(np.eye(3), ("x", "y", "z")):
            arm.add_actuator(name=f"wrist_torque_{name}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT, target="wrist_ball",
                gear=np.concatenate([axis, np.zeros(3)]),
                gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[1] + [0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_NONE, biasprm=[0] * 10,
                ctrllimited=1, ctrlrange=[-4, 4], forcelimited=1, forcerange=[-4, 4])
        base = scene.worldbody.add_frame(name=f"{side}_base", pos=[0, lateral, 0])
        scene.attach(arm, prefix=f"{side}_arm_", frame=base)

        hand = mujoco.MjSpec.from_file(str(hand_path / f"{side}_hand.xml"))
        prefix = f"{side[0]}h_"
        for name in ("WRJ1", "WRJ2"):
            hand.delete(hand.actuator(prefix + "A_" + name))
            hand.delete(hand.joint(prefix + name))
        for part in ("forearm", "wrist"):
            body = hand.body(prefix + part)
            for geom in list(body.geoms):
                hand.delete(geom)
            body.pos = [0, 0, 0]
            body.quat = [1, 0, 0, 0]
            body.mass = .001
            body.ipos = [0, 0, 0]
            body.inertia = [1e-6, 1e-6, 1e-6]
        hand.body(prefix + "palm").pos = [0, 0, .03]
        mount = scene.body(f"{side}_arm_spherical_wrist").add_frame(name=f"{side}_hand_mount")
        scene.attach(hand, prefix=f"{side}_", frame=mount)
        goal = scene.worldbody.add_body(name=f"{side}_goal", mocap=True)
        goal.add_geom(name=f"{side}_goal_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[.009, 0, 0], rgba=[.15, .9, .55, .35], contype=0, conaffinity=0)
        for axis, color in zip(np.eye(3), ([1,.2,.2,.6],[.2,1,.2,.6],[.2,.4,1,.6])):
            goal.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[.002, 0, 0],
                fromto=np.concatenate([np.zeros(3), .045 * axis]), rgba=color,
                contype=0, conaffinity=0)
    for body in scene.bodies:
        if not body.mocap:
            body.gravcomp = 1
    return scene.compile()


class SphericalArmController(ArmController):
    """Five-axis position IK and quaternion-error ball-joint torque control."""

    def __init__(self, model, data):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.joints, self.qids, self.dofs, self.actuators = {}, {}, {}, {}
        self.sites, self.goals, self.ball_bodies = {}, {}, {}
        self.ball_qids, self.ball_dofs, self.ball_acts = {}, {}, {}
        self.targets, self.rotations, self.rotation_targets = {}, {}, {}
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.velocity = np.zeros(6)
        for side in SIDES:
            joints = np.array([model.joint(f"{side}_arm_joint{i}").id for i in range(1, 6)])
            self.joints[side] = joints
            self.qids[side] = model.jnt_qposadr[joints]
            self.dofs[side] = model.jnt_dofadr[joints]
            self.actuators[side] = np.array([model.actuator(f"{side}_arm_actuator{i}").id for i in range(1, 6)])
            joint = model.joint(f"{side}_arm_wrist_ball").id
            self.ball_qids[side] = slice(int(model.jnt_qposadr[joint]), int(model.jnt_qposadr[joint]) + 4)
            self.ball_dofs[side] = slice(int(model.jnt_dofadr[joint]), int(model.jnt_dofadr[joint]) + 3)
            self.ball_acts[side] = np.array([model.actuator(f"{side}_arm_wrist_torque_{axis}").id for axis in "xyz"])
            self.ball_bodies[side] = model.body(f"{side}_arm_spherical_wrist").id
            self.sites[side] = model.site(f"{side}_arm_wrist_site").id
            self.goals[side] = int(model.body_mocapid[model.body(f"{side}_goal").id])
        self.reset(data)

    def reset(self, data):
        mujoco.mj_resetData(self.model, data)
        for side in SIDES:
            data.qpos[self.qids[side]] = HOME[:5]
            data.ctrl[self.actuators[side]] = HOME[:5]
        mujoco.mj_forward(self.model, data)
        self.home, self.home_rotations = {}, {}
        for side in SIDES:
            self.home[side] = data.site_xpos[self.sites[side]].copy()
            self.home_rotations[side] = data.site_xmat[self.sites[side]].reshape(3, 3).copy()
            self.targets[side] = self.home[side].copy()
            self.rotations[side] = self.home_rotations[side].copy()
            self.rotation_targets[side] = self.home_rotations[side].copy()
        self.update_markers(data)

    def solve(self, data, side, dt=.01):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        qids, dofs, acts = self.qids[side], self.dofs[side], self.actuators[side]
        limits = self.model.jnt_range[self.joints[side]]
        self.scratch.qpos[:] = data.qpos
        self.scratch.qpos[qids] = data.ctrl[acts]
        for _ in range(10):
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
            error = self.targets[side] - self.scratch.site_xpos[self.sites[side]]
            mujoco.mj_jacSite(self.model, self.scratch, self.jacp, self.jacr, self.sites[side])
            jacobian = self.jacp[:, dofs]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .005**2 * np.eye(3), error)
            self.scratch.qpos[qids] = np.clip(self.scratch.qpos[qids] + np.clip(delta, -.04, .04),
                                             limits[:, 0] + .01, limits[:, 1] - .01)
        return np.clip(data.ctrl[acts] + np.clip(self.scratch.qpos[qids] - data.ctrl[acts], -.6 * dt, .6 * dt),
                       limits[:, 0] + .01, limits[:, 1] - .01)

    def command(self, data, active=SIDES, dt=.01):
        for side in active:
            delta = rotation_vector(self.rotation_targets[side] @ self.rotations[side].T)
            angle = np.linalg.norm(delta)
            if angle > .8 * dt:
                delta *= .8 * dt / angle
            self.rotations[side] = rotation_matrix(delta) @ self.rotations[side]
            data.ctrl[self.actuators[side]] = self.solve(data, side, dt)
        self.update_markers(data)

    def update_torques(self, data):
        """Call every physics step, including while a stale side holds its pose."""
        for side in SIDES:
            rotation = data.site_xmat[self.sites[side]].reshape(3, 3)
            error = rotation_vector(self.rotations[side] @ rotation.T)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                    self.ball_bodies[side], self.velocity, 0)
            torque = rotation.T @ (12 * error - .8 * self.velocity[:3])
            norm = np.linalg.norm(torque)
            if norm > 4:
                torque *= 4 / norm
            data.ctrl[self.ball_acts[side]] = torque
