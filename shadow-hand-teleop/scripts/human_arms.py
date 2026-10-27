"""Fixed-shoulder human-shaped interaction prototype, not anatomical tracking."""
import mujoco
import mujoco_menagerie as menagerie
import numpy as np

from arm_control import BASE_SCENE, PALM_DOWN_ROTATION, SIDES, rotation_matrix, rotation_vector
from spherical_wrist import SphericalArmController, quaternion


def motor(spec, name, joint, axis, limit):
    spec.add_actuator(name=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=joint, gear=[*axis, 0, 0, 0], gainprm=[1]+[0]*9,
        ctrllimited=1, ctrlrange=[-limit, limit])


def build_human_model(workspace=False):
    from workspace_scene import TABLE_TOP
    scene = mujoco.MjSpec.from_string(BASE_SCENE)
    scene.modelname = "human_shaped_dual_arms"
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.worldbody.add_geom(name="worktop", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[.62, 0, TABLE_TOP - .025], size=[.46, .58, .025], rgba=[.55, .39, .26, 1])
    path = menagerie.get("shadow_hand").path()
    for side, y in (("left", .25), ("right", -.25)):
        upper = scene.worldbody.add_body(name=f"{side}_shoulder", pos=[-.12, y, 1.02])
        upper.add_joint(name=f"{side}_shoulder_ball", type=mujoco.mjtJoint.mjJNT_BALL,
            limited=1, range=[0, 2.3], damping=.2, armature=.01)
        upper.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0,0,0,0,0,-.28], size=[.03,0,0], density=450, rgba=[.65,.68,.72,1])
        fore = upper.add_body(name=f"{side}_elbow", pos=[0,0,-.28])
        fore.add_joint(name=f"{side}_elbow_hinge", type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0,1,0], limited=1, range=[-2.6,-.05], damping=.15, armature=.005)
        fore.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0,0,0,0,0,-.23], size=[.025,0,0], density=450, rgba=[.65,.68,.72,1])
        ball = fore.add_body(name=f"{side}_wrist", pos=[0,0,-.26],
            quat=quaternion(rotation_matrix([0,1.6,0]) @ PALM_DOWN_ROTATION))
        ball.add_joint(name=f"{side}_wrist_ball", type=mujoco.mjtJoint.mjJNT_BALL,
            limited=0, damping=.08, armature=.003)
        ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.018,0,0],
            density=900, rgba=[.2,.24,.29,1])
        ball.add_site(name=f"{side}_wrist_site", size=[.004,0,0])
        for axis, label in zip(np.eye(3), "xyz"):
            motor(scene, f"{side}_shoulder_torque_{label}", f"{side}_shoulder_ball", axis, 18)
            motor(scene, f"{side}_wrist_torque_{label}", f"{side}_wrist_ball", axis, 4)
        scene.add_actuator(name=f"{side}_elbow_position", target=f"{side}_elbow_hinge",
            trntype=mujoco.mjtTrn.mjTRN_JOINT, gainprm=[120]+[0]*9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0,-120,-8]+[0]*7,
            ctrllimited=1,ctrlrange=[-2.6,-.05],forcelimited=1,forcerange=[-18,18])
        hand = mujoco.MjSpec.from_file(str(path / f"{side}_hand.xml"))
        p = f"{side[0]}h_"
        for name in ("WRJ1","WRJ2"):
            hand.delete(hand.actuator(p+"A_"+name)); hand.delete(hand.joint(p+name))
        for name in ("forearm","wrist"):
            body = hand.body(p+name)
            for geom in list(body.geoms):
                hand.delete(geom)
            body.pos=[0,0,0]; body.quat=[1,0,0,0]
            body.mass=.001; body.ipos=[0,0,0]; body.inertia=[1e-6]*3
        hand.body(p+"palm").pos=[0,0,.03]
        scene.attach(hand, prefix=f"{side}_", frame=ball.add_frame())
        goal=scene.worldbody.add_body(name=f"{side}_goal",mocap=True)
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.009,0,0],
            rgba=[.15,.9,.55,.35],contype=0,conaffinity=0)
    for body in scene.bodies:
        if not body.mocap:
            body.gravcomp=1
    if workspace:
        from workspace_scene import add_workspace
        # Props are added AFTER arm gravity compensation: the cup must fall.
        add_workspace(scene)
    return scene.compile()


class HumanArmController(SphericalArmController):
    """Quaternion shoulder IK with physical shoulder/wrist torque servos."""
    # A virtual wrist, not anatomical or constrained by real robot hardware.
    orientation_limit = np.pi
    def __init__(self, model, data):
        self.model=model; self.scratch=mujoco.MjData(model)
        self.sites={}; self.goals={}; self.ball_bodies={}; self.ball_acts={}
        self.shoulder_q={}; self.shoulder_d={}; self.shoulder_b={}; self.shoulder_a={}
        self.elbow_q={}; self.elbow_d={}; self.elbow_a={}; self.arm_targets={}
        self.targets={}; self.rotations={}; self.rotation_targets={}
        self.forearm_targets={s: None for s in SIDES}
        self.forearm_directions={s: None for s in SIDES}
        self.reach_errors={s: 0. for s in SIDES}
        self.jacp=np.zeros((3,model.nv)); self.jacr=np.zeros_like(self.jacp)
        self.velocity=np.zeros(6)
        for s in SIDES:
            j=model.joint(f"{s}_shoulder_ball").id
            q=int(model.jnt_qposadr[j]); d=int(model.jnt_dofadr[j])
            self.shoulder_q[s]=slice(q,q+4); self.shoulder_d[s]=np.arange(d,d+3)
            self.shoulder_b[s]=model.body(f"{s}_shoulder").id
            self.shoulder_a[s]=[model.actuator(f"{s}_shoulder_torque_{a}").id for a in "xyz"]
            j=model.joint(f"{s}_elbow_hinge").id
            self.elbow_q[s]=int(model.jnt_qposadr[j]); self.elbow_d[s]=int(model.jnt_dofadr[j])
            self.elbow_a[s]=model.actuator(f"{s}_elbow_position").id
            self.ball_bodies[s]=model.body(f"{s}_wrist").id
            self.ball_acts[s]=[model.actuator(f"{s}_wrist_torque_{a}").id for a in "xyz"]
            self.sites[s]=model.site(f"{s}_wrist_site").id
            self.goals[s]=int(model.body(f"{s}_goal").mocapid[0])
        self.reset(data)

    def reset(self,data):
        mujoco.mj_resetData(self.model,data)
        self.forearm_targets={s: None for s in SIDES}
        self.forearm_directions={s: None for s in SIDES}
        self.reach_errors={s: 0. for s in SIDES}
        for s in SIDES:
            data.qpos[self.shoulder_q[s]]=quaternion(rotation_matrix([0,-.2,0]))
            data.qpos[self.elbow_q[s]]=-1.4; data.ctrl[self.elbow_a[s]]=-1.4
        mujoco.mj_forward(self.model,data)
        # Same wrist locations, but begin with relaxed, mirrored outward elbows.
        for s in SIDES:
            pose = self.limb_pose(s, data.site_xpos[self.sites[s]].copy())
            data.qpos[self.shoulder_q[s]] = pose[:4]
            data.qpos[self.elbow_q[s]] = pose[4]
            data.ctrl[self.elbow_a[s]] = pose[4]
        mujoco.mj_forward(self.model,data)
        for s in SIDES:
            # Restore palms-down after changing the shoulder's resting swivel.
            rotation=data.site_xmat[self.sites[s]].reshape(3,3)
            joint=self.model.joint(f"{s}_wrist_ball").id
            q=int(self.model.jnt_qposadr[joint])
            data.qpos[q:q+4]=quaternion(rotation.T @ PALM_DOWN_ROTATION)
        mujoco.mj_forward(self.model,data)
        self.home={}; self.home_rotations={}
        for s in SIDES:
            self.arm_targets[s]=data.qpos.copy()
            self.home[s]=data.site_xpos[self.sites[s]].copy()
            self.home_rotations[s]=data.site_xmat[self.sites[s]].reshape(3,3).copy()
            self.targets[s]=self.home[s].copy()
            self.rotations[s]=self.home_rotations[s].copy()
            self.rotation_targets[s]=self.rotations[s].copy()
        self.update_markers(data)

    def forearm_direction(self, data, side):
        from forearm_tracking import unit_vector
        return unit_vector(data.site_xpos[self.sites[side]] - data.body(f"{side}_elbow").xpos)

    def set_forearm_target(self, side, direction):
        from forearm_tracking import unit_vector
        self.forearm_targets[side] = None if direction is None else unit_vector(direction)
        if direction is None:
            self.forearm_directions[side] = None

    def set_target(self, side, position):
        position=np.asarray(position,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():
            raise ValueError("Target must be a finite 3D position")
        # Overlapping hand workspaces: forearms need not remain in separate
        # boxes. Keep the older Panda/spherical workspace limits unchanged.
        lateral=(-.30,.14) if side=="left" else (-.14,.30)
        offset=np.clip(position-self.home[side],[-.16,lateral[0],-.16],[.16,lateral[1],.18])
        self.targets[side]=self.home[side]+offset
        shoulder=self.model.body_pos[self.shoulder_b[side]]
        offset=self.targets[side]-shoulder
        distance=np.linalg.norm(offset)
        # Keep a little elbow bend and avoid a singular fully extended arm.
        if distance > .53:
            self.targets[side]=shoulder+offset*(.53/distance)
        def outward_margin(position):
            pose=self.limb_pose(side,position)
            rotation=np.zeros(9)
            mujoco.mju_quat2Mat(rotation,pose[:4])
            elbow=shoulder+rotation.reshape(3,3) @ np.array([0.,0.,-.28])
            return (1 if side=="left" else -1)*(elbow[1]-shoulder[1])
        if outward_margin(self.targets[side]) < self.elbow_margin(side,self.targets[side])-1e-7:
            # Some far inward reaches cannot satisfy both fixed limb lengths
            # and the outward envelope. Shorten that reach, not the arm bones.
            requested=self.targets[side].copy()
            low,high=0.,1.
            for _ in range(12):
                fraction=(low+high)/2
                candidate=self.home[side]+fraction*(requested-self.home[side])
                if outward_margin(candidate)>=self.elbow_margin(side,candidate):
                    low=fraction
                else:
                    high=fraction
            self.targets[side]=self.home[side]+low*(requested-self.home[side])
        self.reach_errors[side]=float(np.linalg.norm(position-self.targets[side]))

    def elbow_margin(self, side, wrist):
        """Relax shoulder-plane margin near midline, not torso clearance.

        A forearm is only 26 cm long: elbows cannot stay outside 50 cm-wide
        shoulders while the hands meet. Preserve an 8 cm ipsilateral elbow
        clearance instead, with the same outward preference as before.
        """
        sign=1 if side=="left" else -1
        shoulder=self.model.body_pos[self.shoulder_b[side]]
        inward=max(0.,sign*(shoulder[1]-wrist[1])-.14)
        return max(.08-sign*shoulder[1], .015-1.6*inward)

    def limb_pose(self, side, wrist, forearm=None):
        """Two-bone IK with a bounded, outward elbow swivel preference.

        Reach fixes an elbow circle. Forearm input selects a point on that
        circle, within the task-dependent elbow/torso clearance envelope.
        This is a posture envelope, not full collision avoidance.
        """
        from forearm_tracking import unit_vector
        shoulder=self.model.body_pos[self.shoulder_b[side]]
        offset=np.asarray(wrist)-shoulder
        distance=np.clip(np.linalg.norm(offset),.12,.53)
        axis=unit_vector(offset)
        wrist=shoulder+distance*axis
        upper,lower=.28,.26
        along=(upper**2-lower**2+distance**2)/(2*distance)
        center=shoulder+along*axis
        radius=np.sqrt(max(upper**2-along**2,1e-10))
        sign=1 if side=="left" else -1
        preferred=np.array([.06,sign*.09,-.26])
        radial=unit_vector(preferred-axis*(preferred@axis))
        tangent=np.cross(axis,radial)
        angle=0.
        if forearm is not None:
            hint=wrist-lower*unit_vector(forearm)-center
            angle=np.arctan2(hint@tangent,hint@radial)
        lo,hi=-.9,.9
        angle=np.clip(angle,lo,hi)
        # Near the midline, elbows may come inside shoulder width but must
        # remain on their own side of the body centerline.
        a,b=sign*radius*radial[1],sign*radius*tangent[1]
        required=self.elbow_margin(side,wrist)-sign*(center[1]-shoulder[1])
        amplitude=np.hypot(a,b)
        phase=np.arctan2(b,a)
        intervals=[]
        if required<=-amplitude:
            intervals=[(lo,hi)]
        elif required<=amplitude:
            width=np.arccos(np.clip(required/amplitude,-1,1))
            for turn in (-2*np.pi,0,2*np.pi):
                start=max(lo,phase-width+turn);end=min(hi,phase+width+turn)
                if start<=end:
                    intervals.append((start,end))
        if intervals:
            angle=min((np.clip(angle,start,end) for start,end in intervals),key=lambda x:abs(x-angle))
        else:
            # Extreme reaches may make the margin infeasible: choose the most
            # outward point without teleporting or abandoning wrist reach.
            candidates=[lo,hi,np.clip(phase,lo,hi)]
            angle=max(candidates,key=lambda x:a*np.cos(x)+b*np.sin(x))
        elbow=center+radius*(radial*np.cos(angle)+tangent*np.sin(angle))
        upper_direction=unit_vector(elbow-shoulder)
        lower_direction=unit_vector(wrist-elbow)
        y=unit_vector(np.cross(lower_direction,upper_direction))
        z=-upper_direction
        x=np.cross(y,z)
        rotation=np.column_stack((x,y,z))
        return np.r_[quaternion(rotation),-np.arccos(np.clip(upper_direction@lower_direction,-1,1))]

    def command(self,data,active=SIDES,dt=.01):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError("dt must be positive and finite")
        for s in active:
            delta=rotation_vector(self.rotation_targets[s] @ self.rotations[s].T)
            delta*=min(1,2.5*dt/max(np.linalg.norm(delta),1e-12))
            self.rotations[s]=rotation_matrix(delta) @ self.rotations[s]
            if self.forearm_targets[s] is not None:
                if self.forearm_directions[s] is None:
                    self.forearm_directions[s] = self.forearm_direction(data, s)
                # Smooth this optional secondary task instead of snapping elbow.
                current = self.forearm_directions[s]
                target = self.forearm_targets[s]
                from forearm_tracking import unit_vector
                axis = np.cross(current, target)
                angle = np.arctan2(np.linalg.norm(axis), np.clip(current @ target, -1, 1))
                if angle > 1e-8:
                    if np.linalg.norm(axis) < 1e-8:
                        axis = np.cross(current, np.eye(3)[np.argmin(np.abs(current))])
                    current = rotation_matrix(unit_vector(axis)*min(angle,2.5*dt)) @ current
                self.forearm_directions[s] = current
            dofs=np.r_[self.shoulder_d[s],self.elbow_d[s]]
            pose=self.limb_pose(s,self.targets[s],self.forearm_directions[s])
            self.scratch.qpos[:]=self.arm_targets[s]
            self.scratch.qpos[self.shoulder_q[s]]=pose[:4]
            self.scratch.qpos[self.elbow_q[s]]=pose[4]
            v=np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model,v,1,self.arm_targets[s],self.scratch.qpos)
            shoulder_step=v[self.shoulder_d[s]]
            shoulder_step*=min(1,1.8*dt/max(np.linalg.norm(shoulder_step),1e-12))
            v[self.shoulder_d[s]]=shoulder_step
            v[self.elbow_d[s]]=np.clip(v[self.elbow_d[s]],-1.8*dt,1.8*dt)
            v[np.setdiff1d(np.arange(self.model.nv),dofs)]=0
            mujoco.mj_integratePos(self.model,self.arm_targets[s],v,1)
            data.ctrl[self.elbow_a[s]]=self.arm_targets[s][self.elbow_q[s]]
        self.update_markers(data)

    def update_torques(self,data):
        for s in SIDES:
            desired=np.zeros(9)
            mujoco.mju_quat2Mat(desired,self.arm_targets[s][self.shoulder_q[s]])
            current=data.xmat[self.shoulder_b[s]].reshape(3,3)
            error=rotation_vector(desired.reshape(3,3) @ current.T)
            mujoco.mj_objectVelocity(self.model,data,mujoco.mjtObj.mjOBJ_BODY,
                self.shoulder_b[s],self.velocity,0)
            torque=current.T @ (60*error-6*self.velocity[:3])
            torque*=min(1,18/max(np.linalg.norm(torque),1e-12))
            data.ctrl[self.shoulder_a[s]]=torque
        super().update_torques(data)
