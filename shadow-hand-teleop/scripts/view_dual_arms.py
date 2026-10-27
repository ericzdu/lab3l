"""Interactive dual Panda + Shadow Hand demo; optional live Vision Pro input.

Run with mjpython on macOS. In live mode C calibrates wrist position and
orientation. In the --wrist human --workspace scene, E records an episode of
simulated robot state (see episode_recorder.py). No policy or hardware control.
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


# Human-arm workspace views (cup/tray scene, TABLE_TOP 0.60). Tuned so the
# operator view shows both hands, the tray and the cup without the shoulders,
# and the overview shows both whole arms and all props with little empty floor.
WORKSPACE_OPERATOR_VIEW = dict(lookat=(0.42, -0.05, 0.64), distance=0.95, azimuth=0, elevation=-48)
WORKSPACE_OVERVIEW = dict(lookat=(0.22, -0.04, 0.74), distance=1.40, azimuth=210, elevation=-34)


def configure_camera(camera, overview=False, workspace=False):
    """Default viewer/recording camera. workspace=True: human-arm cup/tray scene."""
    if workspace:
        view = WORKSPACE_OVERVIEW if overview else WORKSPACE_OPERATOR_VIEW
        camera.lookat[:] = view["lookat"]
        camera.distance = view["distance"]
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]
    elif overview:
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


def recover_native_reset(data, controller, mapper, last_time, finger_motion=None,
                         recorder=None, auto_recenter=None, assist=None):
    """Undo the MuJoCo window's own Reset (panel button / Backspace).

    That reset calls mj_resetData behind the controller's back: arms drop to
    their raw zero pose (hands below the table) while targets stay stale.
    Detect it by sim time jumping backwards, then do exactly what H does.
    Returns True if a native reset was detected and recovered.
    """
    if not data.time + 1e-9 < last_time:
        return False
    if recorder is not None and recorder.recording:
        # Close the episode on the pre-reset clock so its times stay monotonic.
        reset_time, data.time = data.time, last_time
        if assist:
            # mj_resetData already cleared the weld; record that it ended.
            for kind, payload in assist.release_all(data):
                recorder.event(data, kind, **payload)
        recorder.event(data, "native_reset")
        recorder.stop(data, "reset")
        data.time = reset_time
    reset_scene(data, controller, mapper, finger_motion, assist=assist)
    if auto_recenter:
        auto_recenter.forget()
    return True


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


RECORDING_REQUIRES = "Recording requires --wrist human --workspace"


def toggle_recording(recorder, data, calibrated=True):
    """E: start an episode, or stop and save it. Returns a notice, if any."""
    if recorder is None:
        print(RECORDING_REQUIRES, flush=True)
        return RECORDING_REQUIRES
    if recorder.recording:
        recorder.stop(data, "user_stop")
        return None
    recorder.start(data)
    print(f"Recording episode {recorder.counter:04d}: E stops and saves, X discards.", flush=True)
    if not calibrated:
        recorder.notice = "REC started before calibration (press C)"
    return None


def reset_scene(data, controller, mapper, finger_motion=None, recorder=None, assist=None):
    """H: save any current episode (end_reason "reset"), then reset arms and cup."""
    if assist:
        for kind, payload in assist.release_all(data):
            if recorder is not None:
                recorder.event(data, kind, **payload)
    if recorder is not None and recorder.recording:
        recorder.event(data, "reset")
        recorder.stop(data, "reset")
    if finger_motion:
        finger_motion.clear()
    controller.reset(data)
    if assist:
        assist.forget()
    for side in SIDES:
        mapper.invalidate(side)


def calibrate_side(data, controller, mapper, side, sample, recenter=False):
    """C for one side (or --auto-recenter): anchor tracking at the current wrist pose.

    The robot target becomes the achieved wrist pose and the mapping is
    relative to the current tracked wrist, so the arm cannot jump.
    """
    if recenter and side not in mapper.previous_scales:
        return None, False
    controller.set_target(side, data.site_xpos[controller.sites[side]])
    rotation = data.site_xmat[controller.sites[side]].reshape(3, 3).copy()
    controller.rotations[side] = rotation.copy()
    controller.set_rotation(side, rotation)
    direction = (controller.forearm_direction(data, side)
                 if hasattr(controller, "forearm_direction") else None)
    anchor = mapper.recenter if recenter else mapper.calibrate
    result = anchor(side, sample[2], controller.targets[side], rotation,
                    forearm=sample[3], robot_forearm=direction)
    return direction, result is not False


class AutoRecenter:
    """--auto-recenter: re-anchor a side after a short tracking gap.

    Only a side that was calibrated (following) when its tracking went stale
    qualifies, and only if it is fresh again within `window` seconds.
    Longer gaps, pauses, H and manual C fall back to the default: press C.
    """
    def __init__(self, window=3.):
        if not np.isfinite(window) or window <= 0:
            raise ValueError("Recenter window must be positive and finite")
        self.window = window
        self.lost_at = {}
        self.auto_until = {}

    def update(self, now, fresh, mapper):
        """Call each frame BEFORE stale sides are invalidated; returns [(side, gap)] to recenter."""
        due = []
        for side in SIDES:
            if not fresh[side]:
                if side in mapper.origins:
                    self.lost_at[side] = now
            elif side in self.lost_at:
                gap = now - self.lost_at.pop(side)
                if gap <= self.window and side not in mapper.origins:
                    due.append((side, gap))
        return due

    def recentered(self, side, now):
        self.auto_until[side] = now + 1.

    def showing(self, side, now):
        return self.auto_until.get(side, -np.inf) > now

    def forget(self):
        self.lost_at.clear()


def headset_lines(fresh, mapper, controller, paused=False, error=None, recorder=None,
                  data=None, auto_recenter=None, now=None):
    """Short status for the headset overlay: per-side tracking/follow, REC, last outcome."""
    if error:
        first = "Connection error"
    elif paused:
        first = "PAUSED (Space resumes, then C)"
    else:
        parts = []
        for side in SIDES:
            follow = "FOLLOW" if side in mapper.origins else "press C"
            if (follow == "FOLLOW" and auto_recenter is not None and now is not None
                    and auto_recenter.showing(side, now)):
                follow += " (auto)"
            reach = " REACH" if getattr(controller, "reach_errors", {}).get(side, 0) > .005 else ""
            parts.append(f"{side[0].upper()}: {'LIVE' if fresh[side] else 'WAIT'} · {follow}{reach}")
        first = "   ".join(parts)
    lines = [first]
    if recorder is not None:
        if recorder.recording:
            seconds = recorder.elapsed(data)
            lines.append(f"\u25cf REC {int(seconds // 60):02d}:{int(seconds % 60):02d}")
        if recorder.last_saved:
            lines.append("Last: " + recorder.outcome_text())
    return lines


def video_size(text):
    try:
        width, height = (int(value) for value in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, for example 640x480")
    if not (16 <= width <= 4096 and 16 <= height <= 4096) or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("video size must be even, 16-4096 pixels per side")
    return width, height


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
    parser.add_argument("--auto-recenter", action="store_true",
                        help="Live only: after a tracking gap shorter than --recenter-window, re-anchor "
                             "that (previously calibrated) arm instead of waiting for C (default: off)")
    parser.add_argument("--recenter-window", type=float, default=3.0,
                        help="Seconds a gap may last for --auto-recenter (default 3.0)")
    parser.add_argument("--grasp-assist", action="store_true",
                        help="--workspace only: weld a closed hand (thumb + a finger touching) to the "
                             "cup until it opens; recorded and flagged in episodes (default: off)")
    headset = parser.add_argument_group("headset view (live only): sim view as a 2D video window in Vision Pro")
    headset.add_argument("--headset-view", action="store_true", help="Stream the sim view to the headset")
    # 960x540 @ 20 fps: 1280x720 @ 30 goes over the render budget on about
    # half its frames with recording on (README "Headset view").
    headset.add_argument("--headset-size", type=video_size, default=(960, 540),
                         help="Streamed WIDTHxHEIGHT (default 960x540)")
    headset.add_argument("--headset-fps", type=float, default=20, help="Target stream rate, wall clock (default 20)")
    headset.add_argument("--headset-camera", choices=("follow", "operator", "overview"), default="follow",
                         help="follow = whatever the Mac viewer shows (default); operator/overview = fixed views")
    headset.add_argument("--headset-budget-ms", type=float, default=16,
                         help="A headset frame costing more than this skips the next frame slot (default 16)")
    recording = parser.add_argument_group("episode recording (--wrist human --workspace only)")
    recording.add_argument("--record-dir", default=None,
                           help="Episode folder (default: data/episodes in the project)")
    recording.add_argument("--task", default="cup_to_tray", help="Task id stored in metadata")
    recording.add_argument("--operator", default="", help="Optional operator tag, e.g. initials")
    recording.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                           help="Render operator/overview camera videos (default: on)")
    recording.add_argument("--video-fps", type=float, default=30, help="Video frame rate in sim time")
    recording.add_argument("--video-size", type=video_size, default=(640, 480),
                           help="Per-camera WIDTHxHEIGHT (default 640x480)")
    recording.add_argument("--max-episode-seconds", type=float, default=120,
                           help="Auto-stop and save after this much sim time")
    recording.add_argument("--no-auto-label", action="store_true",
                           help="Do not label episodes after saving")
    args = parser.parse_args()
    if args.workspace and args.wrist != "human":
        parser.error("--workspace currently requires --wrist human")
    if args.address and (args.headless or args.demo):
        parser.error("Use live tracking or the offline demo, not both")
    if args.seconds < 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be finite and nonnegative")
    if not np.isfinite(args.recenter_window) or args.recenter_window <= 0:
        parser.error("--recenter-window must be positive and finite")
    if args.headset_view and not args.address:
        parser.error("--headset-view needs live tracking (--address)")
    if not np.isfinite(args.headset_fps) or not 0 < args.headset_fps <= 60:
        parser.error("--headset-fps must be in (0, 60]")
    if not np.isfinite(args.headset_budget_ms) or args.headset_budget_ms <= 0:
        parser.error("--headset-budget-ms must be positive and finite")
    if args.grasp_assist and not args.workspace:
        parser.error("--grasp-assist needs the cup (--workspace)")
    if args.auto_recenter and not args.address:
        parser.error("--auto-recenter needs live tracking (--address)")
    if len(args.operator) > 32:
        parser.error("--operator must be at most 32 characters")
    from episode_recorder import TASK_PATTERN
    if not TASK_PATTERN.match(args.task):
        parser.error("--task must be 1-40 letters, digits, '_' or '-'")
    for name in ("video_fps", "max_episode_seconds"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    return args


def main():
    args = parse_args()
    mapper = WristMapper(args.scale)
    model, data, controller, hands = setup(args.wrist, workspace=args.workspace)
    finger_motion=FingerTransition(model) if not args.address else None
    assist = None
    if args.grasp_assist:
        from grasp_assist import GraspAssist
        assist = GraspAssist(model, hands)
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

    headset = None
    if args.headset_view:
        from headset_view import HeadsetView
        headset = HeadsetView(model, size=args.headset_size, fps=args.headset_fps,
                              camera=args.headset_camera, budget=args.headset_budget_ms / 1000)

    bridge = None
    if args.address:
        # The SDK installs signal handlers; import and construct on main thread.
        from avp_stream import VisionProStreamer
        from teleop_shadow_hands import TrackingBridge

        # Outbound video is set up on the bridge thread once the streamer
        # exists; HeadsetView.attach starts WebRTC on its own thread.
        bridge = TrackingBridge(VisionProStreamer, args.address, with_wrist=True,
                                on_streamer=headset.attach if headset else None)

    recorder = None
    if args.workspace:
        from episode_recorder import DEFAULT_RECORD_DIR, EpisodeRecorder
        recorder = EpisodeRecorder(model, controller, hands,
            record_dir=args.record_dir or DEFAULT_RECORD_DIR, task=args.task,
            operator=args.operator,
            source="live" if bridge else "scripted_demo" if args.demo else "keyboard",
            wrist_mode=args.wrist, workspace=args.workspace, scale=args.scale,
            video=args.video, video_fps=args.video_fps, video_size=args.video_size,
            max_seconds=args.max_episode_seconds, auto_label=not args.no_auto_label,
            assist=assist)
    record_notice = None
    end_reason = "viewer_closed"
    auto_recenter = AutoRecenter(args.recenter_window) if bridge and args.auto_recenter else None

    events = SimpleQueue()
    selected, paused, overview = "right", False, False
    last_status = last_panel_status = None
    print(f"Two arms + Shadow Hands; wrist assembly: {args.wrist}.", flush=True)
    print("Use buttons/keyboard in the separate 'Holos — Hand controls' window.", flush=True)
    print("The MuJoCo window keeps its native camera/display shortcuts; robot keys are NOT bound there.", flush=True)
    print("1: select LEFT | 2: select RIGHT | W/S: forward/back | A/D: left/right | R/V: up/down", flush=True)
    print("H: reset home | SPACE: pause/resume", flush=True)
    print("B: switch operator / overview camera. Default view looks forward over palms-down hands.", flush=True)
    if recorder:
        print("E: start/stop+save an episode | X: discard it. Episodes are simulated robot state.", flush=True)
    if bridge:
        print("LIVE: face forward, hold hands comfortably, then press C in Hand controls.", flush=True)
        print("C: calibrate/recenter wrist position AND orientation. Fingers use the previous mapping.", flush=True)
        print("After a tracking interruption or pause, press C again." +
              (f" Auto-recenter: gaps under {args.recenter_window:g} s re-anchor by themselves."
               if auto_recenter else ""), flush=True)
        if headset:
            print(f"Headset view: streaming the {args.headset_camera} camera at "
                  f"{args.headset_size[0]}x{args.headset_size[1]}, {args.headset_fps:g} fps "
                  "(2D video window, not AR). It appears once the headset connects.", flush=True)
        if args.wrist == "human":
            print("Optional 27-point forearm tracking: panel shows ON / press C / unavailable.", flush=True)
            print("Forearm guides elbow posture; shoulder is fixed, not full-body tracking.", flush=True)
        bridge.thread.start()
    else:
        print("F: both fists (0.9 s) | O: both open gradually (1.3 s)", flush=True)
        print("I/K: bend wrist up/down | J/L: tilt left/right | U/P: turn palm about forearm", flush=True)
        print("Offline keyboard mode. Vision Pro is not required." +
              (" Press E to record an episode." if recorder else " No data is recorded."), flush=True)
    started = time.monotonic()
    panel = None
    quit_requested = False
    try:
        # The stock viewer processes its shortcuts before its Python callback.
        # Keep all robot input in the separate control window instead.
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                configure_camera(viewer.cam, workspace=args.workspace)
            from control_panel import ControlPanel
            panel = ControlPanel(events, live=bridge is not None, recording=recorder is not None)
            physics_clock = PhysicsClock(model.opt.timestep)
            last_tick = time.monotonic()
            last_sim_time = data.time
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
                    if recover_native_reset(data, controller, mapper, last_sim_time,
                                            finger_motion, recorder, auto_recenter, assist):
                        print("MuJoCo window Reset detected: arms and cup returned home"
                              + ("; live arms need C" if bridge else "")
                              + ". Prefer Home · H in Hand controls.", flush=True)
                    due = (auto_recenter.update(frame_start, fresh, mapper)
                           if auto_recenter and not paused and not error else [])
                    # Reacquisition must not produce a jump from an old origin.
                    for side in SIDES:
                        if bridge and not fresh[side]:
                            mapper.invalidate(side)
                        if bridge and hasattr(controller, "set_forearm_target"):
                            # No current direction -> ordinary wrist-only IK.
                            controller.set_forearm_target(side, None if not fresh[side] else
                                mapper.forearm_target(side, samples[side][3]))
                    for side, gap in due:
                        # Default rule is "press C"; this is the opt-in shortcut.
                        if calibrate_side(data, controller, mapper, side, samples[side], recenter=True)[1]:
                            auto_recenter.recentered(side, frame_start)
                            print(f"Auto-recentered {side} wrist after a {gap:.1f} s tracking gap", flush=True)
                            if recorder and recorder.recording:
                                recorder.event(data, "auto_recenter", side=side, gap_s=round(gap, 3))
                    while True:
                        try:
                            key = events.get_nowait()
                        except Empty:
                            break
                        if key == glfw.KEY_ESCAPE:
                            quit_requested = True
                        elif key == glfw.KEY_B:
                            overview = not overview
                            configure_camera(viewer.cam, overview=overview, workspace=args.workspace)
                        elif key in (glfw.KEY_1, glfw.KEY_2):
                            selected = "left" if key == glfw.KEY_1 else "right"
                            print(f"Selected {selected} arm", flush=True)
                        elif key == glfw.KEY_SPACE:
                            paused = not paused
                            for side in SIDES:
                                mapper.invalidate(side)
                            if auto_recenter:
                                auto_recenter.forget()
                            if recorder:
                                recorder.event(data, "pause" if paused else "resume")
                            print("PAUSED" if paused else "RESUMED (live arms need C)", flush=True)
                        elif key == glfw.KEY_H:
                            reset_scene(data, controller, mapper, finger_motion, recorder, assist)
                            if auto_recenter:
                                auto_recenter.forget()
                            print("Reset to home; live arms need C", flush=True)
                        elif key == glfw.KEY_E:
                            record_notice = toggle_recording(recorder, data,
                                calibrated=not bridge or all(side in mapper.origins for side in SIDES))
                        elif key == glfw.KEY_X:
                            if recorder:
                                recorder.discard()
                        elif key == glfw.KEY_C and bridge and not paused:
                            calibrated=[]
                            if auto_recenter:
                                auto_recenter.forget()
                            for side in SIDES:
                                if fresh[side]:
                                    direction, _ = calibrate_side(data, controller, mapper, side, samples[side])
                                    calibrated.append(side)
                                    print(f"Calibrated {side} wrist", flush=True)
                                    if direction is not None:
                                        print(f"{side} forearm: " + ("calibrated" if side in mapper.forearm_alignments
                                            else "unavailable; wrist-only fallback"), flush=True)
                                else:
                                    print(f"Cannot calibrate {side}: waiting for fresh tracking", flush=True)
                            hand_span=None
                            if args.wrist=="human" and len(calibrated)==2:
                                hand_span=mapper.calibrate_hand_span()
                                if hand_span:
                                    print("Both-hand span calibrated: lateral closing follows your initial hand spacing.",flush=True)
                                else:
                                    print("Hand-span calibration skipped: hold hands 12–85 cm apart, face forward, and press C.",flush=True)
                            if recorder:
                                from episode_recorder import mapper_state
                                recorder.event(data, "calibrate", sides=calibrated, hand_span=hand_span,
                                               mapper=mapper_state(mapper, calibrated))
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
                        for kind, payload in assist.update(data) if assist else ():
                            print(f"Grasp assist {'ON' if kind == 'assist_on' else 'OFF'} ({payload['side']}"
                                  + (f", {payload['reason']}" if "reason" in payload else "") + ")", flush=True)
                            if recorder:
                                recorder.event(data, kind, **payload)
                        if recorder and recorder.recording:
                            recorder.sample(data, samples=samples, fresh=fresh, paused=paused, mapper=mapper)
                    else:
                        controller.update_markers(data)
                    if not np.isfinite(data.qpos).all():
                        raise RuntimeError("Non-finite simulation state; stopping")
                    if recorder and recorder.recording:
                        recorder.watch(data, fresh)
                        recorder.maybe_render(data)
                        if recorder.expired(data):
                            recorder.stop(data, "max_duration")
                if bridge:
                    status = " | ".join(
                        f"{side}: {'LIVE' if fresh[side] else 'WAIT / HOLD'}, "
                        f"arm {'FOLLOW' if side in mapper.origins else 'press C'}" +
                        (" (auto)" if auto_recenter and side in mapper.origins
                            and auto_recenter.showing(side, frame_start) else "") +
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
                # The REC timer updates the panel only, not the terminal.
                if status != last_status:
                    print(status, flush=True)
                    last_status = status
                note = recorder.status_text(data) if recorder else record_notice
                panel_status = status + (" · " + note if note else "")
                if panel_status != last_panel_status:
                    panel.status(panel_status)
                    last_panel_status = panel_status
                if headset:
                    lines = headset_lines(fresh, mapper, controller, paused, error, recorder,
                                          data, auto_recenter, frame_start)
                    with viewer.lock():
                        headset.maybe_send(data, viewer.cam, lines,
                            reuse=recorder.latest_frame("operator") if recorder else None)
                    for message in headset.pop_messages():
                        print(message, flush=True)
                last_sim_time = data.time
                viewer.sync()
                time.sleep(max(0, 0.01 - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if recorder and recorder.recording:
            recorder.event(data, "error", message=f"{type(exc).__name__}: {exc}")
            end_reason = "error"
        raise
    finally:
        if recorder and recorder.recording:
            try:
                recorder.stop(data, end_reason)
            except Exception as exc:
                from episode_recorder import scrub
                print(f"Could not save the current episode: {scrub(exc)}", flush=True)
        if recorder:
            recorder.close()
        if headset:
            headset.close()
        if panel:
            panel.close()
        if bridge:
            bridge.close()
        if args.screenshot:
            save_screenshot(model, data, args.screenshot)
        if recorder and recorder.saved:
            print(f"Stopped. {len(recorder.saved)} episode(s) saved this session.", flush=True)
        else:
            print("Stopped. No data saved (except an explicitly requested screenshot).", flush=True)


if __name__ == "__main__":
    main()
