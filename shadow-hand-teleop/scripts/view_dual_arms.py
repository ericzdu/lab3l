"""Interactive dual Panda + Shadow Hand demo; optional live Vision Pro input.

Run with mjpython on macOS. In live mode C calibrates wrist position and
orientation. No recording, policy, or hardware control.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from arm_control import (ArmController, SIDES, WristMapper, build_arm_model,
                         rotation_matrix, rotation_vector)
from hand_retargeting import actuator_indices, apply_targets


def setup(wrist_mode="spherical", workspace=False):
    if wrist_mode == "human":
        from human_arms import HumanArmController, build_human_model
        model = build_human_model(workspace=workspace)
        controller_type = HumanArmController
    elif wrist_mode == "spherical":
        from spherical_wrist import SphericalArmController, build_spherical_model
        model = build_spherical_model()
        controller_type = SphericalArmController
    else:
        model = build_arm_model()
        controller_type = ArmController
    data = mujoco.MjData(model)
    controller = controller_type(model, data)
    hands = {side: actuator_indices(model, side, include_wrist=wrist_mode == "panda") for side in SIDES}
    return model, data, controller, hands


def fist_targets(indices, closed):
    # Named finger targets work with either the original 20-actuator hand or
    # the 18-actuator fingers on the new ball joint. Wrist commands stay intact.
    values = {name: 0.0 for name in indices if name not in ("WRJ1", "WRJ2")}
    if closed:
        values.update(THJ4=1.0, THJ2=.5, THJ1=1.2, LFJ5=.5)
        for prefix in ("FF", "MF", "RF", "LF"):
            values[prefix + "J3"] = 1.3
            values[prefix + "J0"] = 2.5
    return values


def set_fist(model, data, indices, closed):
    # Immediate helper retained for offline physics tests, not viewer input.
    apply_targets(data.ctrl, fist_targets(indices,closed), indices, model.actuator_ctrlrange)


class FingerTransition:
    """Cubic eased open/close commands, advanced at the physics timestep.

    Only actuator commands change: no finger or cup state is teleported.
    Repeated key events do not restart the same transition.
    """
    def __init__(self, model, open_seconds=1.3, close_seconds=.9):
        if any(not np.isfinite(value) or value<=0 for value in (open_seconds,close_seconds)):
            raise ValueError('Transition durations must be positive and finite')
        self.model=model
        self.open_seconds=open_seconds
        self.close_seconds=close_seconds
        self.tasks={}

    def request(self, data, indices, closed):
        names=fist_targets(indices,closed)
        ids=np.array([indices[name] for name in names])
        goal=np.clip([names[name] for name in names],
            self.model.actuator_ctrlrange[ids,0],self.model.actuator_ctrlrange[ids,1])
        key=tuple(ids)
        if key in self.tasks and np.array_equal(self.tasks[key]['goal'],goal):
            return
        if np.allclose(data.ctrl[ids],goal,atol=1e-12,rtol=0):
            self.tasks.pop(key,None)
            return
        self.tasks[key]=dict(ids=ids,start=data.ctrl[ids].copy(),goal=goal,elapsed=0.,
            duration=self.close_seconds if closed else self.open_seconds)

    def advance(self, data, dt):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('dt must be positive and finite')
        for key,task in list(self.tasks.items()):
            task['elapsed']+=dt
            fraction=min(1.,task['elapsed']/task['duration'])
            eased=fraction*fraction*(3-2*fraction)
            data.ctrl[task['ids']]=task['start']+eased*(task['goal']-task['start'])
            if fraction>=1:
                self.tasks.pop(key)

    def clear(self):
        self.tasks.clear()


def advance_physics(model, data, controller, steps=5, finger_motion=None):
    update_torques = getattr(controller, "update_torques", None)
    for _ in range(steps):
        if finger_motion is not None:
            finger_motion.advance(data,model.opt.timestep)
        if update_torques:
            update_torques(data)
        mujoco.mj_step(model, data)


class PhysicsClock:
    """Render-rate-independent physics budget; never replay a long stall."""
    def __init__(self, timestep):
        self.timestep = timestep
        self.remainder = 0.

    def steps(self, elapsed, running=True):
        if not running:
            self.remainder = 0.
            return 0
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed must be finite and nonnegative")
        self.remainder += min(elapsed, .05)
        steps = int((self.remainder + 1e-12) / self.timestep)
        self.remainder = max(0., self.remainder - steps * self.timestep)
        return steps


def configure_camera(camera, overview=False):
    if overview:
        camera.lookat[:] = [0.35, 0, 0.46]
        camera.distance = 2.30
        camera.azimuth = 150
        camera.elevation = -22
    else:
        # Behind and above the hands, looking forward/down like an operator
        # at a workbench. World +Y (robot left) appears on screen left.
        camera.lookat[:] = [0.55, 0, 0.62]
        camera.distance = 0.95
        camera.azimuth = 0
        camera.elevation = -32


def save_screenshot(model, data, destination):
    from PIL import Image

    camera = mujoco.MjvCamera()
    configure_camera(camera)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 900
    with mujoco.Renderer(model, height=900, width=1280) as renderer:
        renderer.update_scene(data, camera=camera)
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).save(output)
    print(f"Screenshot saved: {output}", flush=True)


def demo_targets(controller, t):
    # Small smooth relative movements to exercise actual arm joint control.
    for side, sign in (("left", 1), ("right", -1)):
        controller.set_target(side, controller.home[side] + np.array([
            0.04 * np.sin(t), sign * 0.025 * np.sin(t), 0.04 * (1 - np.cos(t))]))
        controller.set_rotation(side, rotation_matrix([
            sign * 0.18 * np.sin(t), 0.12 * np.sin(0.7 * t), 0.08 * np.sin(0.6 * t)
        ]) @ controller.home_rotations[side])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="Current Tracking Streamer IP or remote room code")
    parser.add_argument("--workspace", action="store_true", help="Table, movable cup and fixed tray (human arms)")
    parser.add_argument("--scale", type=float, default=0.6, help="Robot travel / human wrist travel (default 0.6)")
    parser.add_argument("--wrist", choices=("spherical", "panda", "human"), default="spherical",
                        help="Simplified ball-joint wrists (default), or previous Panda assembly")
    parser.add_argument("--seconds", type=float, default=0, help="Optional viewer time limit")
    parser.add_argument("--demo", action="store_true", help="Offline scripted arm motion, not human data")
    parser.add_argument("--headless", action="store_true", help="Run offline physics without an interactive viewer")
    parser.add_argument("--screenshot", help="Save a PNG at exit")
    args = parser.parse_args()
    if args.workspace and args.wrist != "human":
        parser.error("--workspace currently requires --wrist human")
    if args.address and (args.headless or args.demo):
        parser.error("Use live tracking or the offline demo, not both")
    if args.seconds < 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    mapper = WristMapper(args.scale)
    model, data, controller, hands = setup(args.wrist, workspace=args.workspace)
    finger_motion=FingerTransition(model) if not args.address else None
    if args.headless:
        duration = args.seconds or 4
        for _ in range(max(1, int(duration / 0.01))):
            if args.demo:
                demo_targets(controller, data.time)
            controller.command(data)
            advance_physics(model, data, controller)
        if not np.isfinite(data.qpos).all() or max(abs(data.qvel)) > 100:
            raise RuntimeError("Offline physics check failed")
        print(f"Offline arm simulation passed: {data.time:.2f}s, {model.nu} actuators, {data.ncon} contacts")
        for side in SIDES:
            error = np.linalg.norm(data.site_xpos[controller.sites[side]] - controller.targets[side])
            print(f"{side} wrist target error: {error * 1000:.2f} mm")
            actual = data.site_xmat[controller.sites[side]].reshape(3, 3)
            angle = np.linalg.norm(rotation_vector(controller.rotation_targets[side] @ actual.T))
            print(f"{side} orientation target error: {np.rad2deg(angle):.2f} deg")
        if args.screenshot:
            save_screenshot(model, data, args.screenshot)
        return

    bridge = None
    if args.address:
        # The SDK installs signal handlers; import and construct on main thread.
        from avp_stream import VisionProStreamer
        from teleop_shadow_hands import TrackingBridge

        bridge = TrackingBridge(VisionProStreamer, args.address, with_wrist=True)

    events = SimpleQueue()
    selected, paused, overview = "right", False, False
    last_status = None
    print(f"Two arms + Shadow Hands; wrist assembly: {args.wrist}.", flush=True)
    print("Use buttons/keyboard in the separate 'Holos — Hand controls' window.", flush=True)
    print("The MuJoCo window keeps its native camera/display shortcuts; robot keys are NOT bound there.", flush=True)
    print("1: select LEFT | 2: select RIGHT | W/S: forward/back | A/D: left/right | R/V: up/down", flush=True)
    print("H: reset home | SPACE: pause/resume", flush=True)
    print("B: switch operator / overview camera. Default view looks forward over palms-down hands.", flush=True)
    if bridge:
        print("LIVE: face forward, hold hands comfortably, then press C in Hand controls.", flush=True)
        print("C: calibrate/recenter wrist position AND orientation. Fingers use the previous mapping.", flush=True)
        print("After a tracking interruption or pause, press C again. No data recording is enabled.", flush=True)
        if args.wrist == "human":
            print("Optional 27-point forearm tracking: panel shows ON / press C / unavailable.", flush=True)
            print("Forearm guides elbow posture; shoulder is fixed, not full-body tracking.", flush=True)
        bridge.thread.start()
    else:
        print("F: both fists (0.9 s) | O: both open gradually (1.3 s)", flush=True)
        print("I/K: bend wrist up/down | J/L: tilt left/right | U/P: turn palm about forearm", flush=True)
        print("Offline keyboard mode. Vision Pro is not required. No data is recorded.", flush=True)
    started = time.monotonic()
    panel = None
    quit_requested = False
    try:
        # The stock viewer processes its shortcuts before its Python callback.
        # Keep all robot input in the separate control window instead.
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                configure_camera(viewer.cam)
            from control_panel import ControlPanel
            panel = ControlPanel(events, live=bridge is not None)
            physics_clock = PhysicsClock(model.opt.timestep)
            last_tick = time.monotonic()
            while viewer.is_running():
                frame_start = time.monotonic()
                elapsed = frame_start - last_tick
                last_tick = frame_start
                if args.seconds and frame_start - started >= args.seconds:
                    break
                samples, error = bridge.snapshot() if bridge else ({}, None)
                fresh = {side: side in samples and frame_start - samples[side][0] < 0.5
                         and not error for side in SIDES}
                with viewer.lock():
                    # Reacquisition must not produce a jump from an old origin.
                    for side in SIDES:
                        if bridge and not fresh[side]:
                            mapper.invalidate(side)
                        if bridge and hasattr(controller, "set_forearm_target"):
                            # No current direction -> ordinary wrist-only IK.
                            controller.set_forearm_target(side, None if not fresh[side] else
                                mapper.forearm_target(side, samples[side][3]))
                    while True:
                        try:
                            key = events.get_nowait()
                        except Empty:
                            break
                        if key == glfw.KEY_ESCAPE:
                            quit_requested = True
                        elif key == glfw.KEY_B:
                            overview = not overview
                            configure_camera(viewer.cam, overview=overview)
                        elif key in (glfw.KEY_1, glfw.KEY_2):
                            selected = "left" if key == glfw.KEY_1 else "right"
                            print(f"Selected {selected} arm", flush=True)
                        elif key == glfw.KEY_SPACE:
                            paused = not paused
                            for side in SIDES:
                                mapper.invalidate(side)
                            print("PAUSED" if paused else "RESUMED (live arms need C)", flush=True)
                        elif key == glfw.KEY_H:
                            if finger_motion:
                                finger_motion.clear()
                            controller.reset(data)
                            for side in SIDES:
                                mapper.invalidate(side)
                            print("Reset to home; live arms need C", flush=True)
                        elif key == glfw.KEY_C and bridge and not paused:
                            calibrated=[]
                            for side in SIDES:
                                if fresh[side]:
                                    controller.set_target(side, data.site_xpos[controller.sites[side]])
                                    rotation = data.site_xmat[controller.sites[side]].reshape(3, 3).copy()
                                    controller.rotations[side] = rotation.copy()
                                    controller.set_rotation(side, rotation)
                                    direction = (controller.forearm_direction(data, side)
                                        if hasattr(controller, "forearm_direction") else None)
                                    mapper.calibrate(side, samples[side][2], controller.targets[side], rotation,
                                        forearm=samples[side][3], robot_forearm=direction)
                                    calibrated.append(side)
                                    print(f"Calibrated {side} wrist", flush=True)
                                    if direction is not None:
                                        print(f"{side} forearm: " + ("calibrated" if side in mapper.forearm_alignments
                                            else "unavailable; wrist-only fallback"), flush=True)
                                else:
                                    print(f"Cannot calibrate {side}: waiting for fresh tracking", flush=True)
                            if args.wrist=="human" and len(calibrated)==2:
                                if mapper.calibrate_hand_span():
                                    print("Both-hand span calibrated: lateral closing follows your initial hand spacing.",flush=True)
                                else:
                                    print("Hand-span calibration skipped: hold hands 12–85 cm apart, face forward, and press C.",flush=True)
                        elif key in (glfw.KEY_F, glfw.KEY_O) and not bridge:
                            for side in SIDES:
                                finger_motion.request(data, hands[side], key == glfw.KEY_F)
                        elif not bridge and not paused:
                            rotation_axis = {
                                glfw.KEY_I: [0, -1, 0], glfw.KEY_K: [0, 1, 0],
                                glfw.KEY_J: [0, 0, 1], glfw.KEY_L: [0, 0, -1],
                                glfw.KEY_U: [1, 0, 0], glfw.KEY_P: [-1, 0, 0],
                            }.get(key)
                            if rotation_axis is not None:
                                controller.set_rotation(selected,
                                    rotation_matrix(np.deg2rad(5) * np.array(rotation_axis))
                                    @ controller.rotation_targets[selected])
                            direction = {
                                glfw.KEY_W: [1, 0, 0], glfw.KEY_S: [-1, 0, 0],
                                glfw.KEY_A: [0, 1, 0], glfw.KEY_D: [0, -1, 0],
                                glfw.KEY_R: [0, 0, 1], glfw.KEY_V: [0, 0, -1],
                            }.get(key)
                            if direction is not None:
                                controller.set_target(selected, controller.targets[selected] + 0.015 * np.array(direction))
                    if quit_requested:
                        break
                    if args.demo and not paused:
                        demo_targets(controller, data.time)
                    steps = physics_clock.steps(elapsed, running=
                        not paused and (not bridge or any(fresh.values())) and not error)
                    control_dt = steps * model.opt.timestep
                    active = list(SIDES) if not bridge else []
                    if bridge and not paused and not error:
                        for side in SIDES:
                            if fresh[side]:
                                desired = data.ctrl.copy()
                                fingers = {name: value for name, value in samples[side][1].items()
                                           if name not in ("WRJ1", "WRJ2")}
                                apply_targets(desired, fingers, hands[side], model.actuator_ctrlrange)
                                ids = [idx for name, idx in hands[side].items() if name not in ("WRJ1", "WRJ2")]
                                data.ctrl[ids] += np.clip(desired[ids] - data.ctrl[ids], -7.5*control_dt, 7.5*control_dt)
                                target = mapper.target(side, samples[side][2])
                                if target is not None:
                                    controller.set_target(side, target)
                                    controller.set_rotation(side, mapper.rotation_target(side, samples[side][2]))
                                    if hasattr(controller, "set_forearm_target"):
                                        controller.set_forearm_target(side, mapper.forearm_target(side, samples[side][3]))
                                    active.append(side)
                    if steps:
                        controller.command(data, active=active, dt=control_dt)
                        advance_physics(model, data, controller, steps=steps, finger_motion=finger_motion)
                    else:
                        controller.update_markers(data)
                    if not np.isfinite(data.qpos).all():
                        raise RuntimeError("Non-finite simulation state; stopping")
                if bridge:
                    status = " | ".join(
                        f"{side}: {'LIVE' if fresh[side] else 'WAIT / HOLD'}, "
                        f"arm {'FOLLOW' if side in mapper.origins else 'press C'}" +
                        ((", forearm " + ("unavailable" if not fresh[side] or samples[side][3] is None
                            else "ON" if side in mapper.forearm_alignments else "press C"))
                            if args.wrist == "human" else "") +
                        (", REACH LIMIT" if getattr(controller,'reach_errors',{}).get(side,0)>.005 else "")
                        for side in SIDES)
                    if paused:
                        status = "PAUSED"
                    if error:
                        status = "Connection error: " + error
                else:
                    status = f"Offline · {selected.title()} arm · {'PAUSED' if paused else 'Ready'}"
                    if finger_motion.tasks:
                        status += ' · Gradual hand motion'
                    if getattr(controller,'reach_errors',{}).get(selected,0)>.005:
                        status += ' · REACH LIMIT: move target closer'
                if status != last_status:
                    print(status, flush=True)
                    panel.status(status)
                    last_status = status
                viewer.sync()
                time.sleep(max(0, 0.01 - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        if panel:
            panel.close()
        if bridge:
            bridge.close()
        if args.screenshot:
            save_screenshot(model, data, args.screenshot)
        print("Stopped. No data saved (except an explicitly requested screenshot).", flush=True)


if __name__ == "__main__":
    main()
