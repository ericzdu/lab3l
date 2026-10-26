"""Record episodes from the human-arm cup/tray workspace to versioned HDF5.

An episode is SIMULATED robot state driven by human tracking (live), keyboard
targets or scripted targets. It is not real-robot data. Raw Vision Pro input
is stored next to the retargeted commands so old episodes can be re-run
through future retargeters. See README "Recording episodes" for the layout.

Buffers live in memory until stop(); a 120 s episode is well under 100 MB.
Never store the Tracking Streamer address, usernames or absolute paths here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import mujoco
import numpy as np

from arm_control import SIDES
from hand_retargeting import FINGER_TARGET_NAMES


SCHEMA = "lab3l.episode"
SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_DIR = PROJECT_ROOT / "data" / "episodes"
SOURCES = ("live", "keyboard", "scripted_demo")
END_REASONS = ("user_stop", "reset", "viewer_closed", "max_duration", "error")
CONTACT_GROUPS = ("thumb", "ff", "mf", "rf", "lf", "palm", "other_hand")
TASK_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
EPISODE_PATTERN = re.compile(r"^\d{8}-\d{6}_.+_(\d{4,})$")


def relative_path(path):
    """Project-relative (or home-relative) display path; never absolute."""
    path = Path(path).resolve()
    for root, prefix in ((PROJECT_ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def scrub(text):
    """Remove local path prefixes from free text (exception messages)."""
    text = str(text)
    for root, replacement in ((str(PROJECT_ROOT), "."), (str(Path.home()), "~")):
        text = text.replace(root, replacement)
    return text


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def mapper_state(mapper, sides):
    """JSON-safe snapshot of WristMapper calibration for the given sides."""
    fields = ("origins", "anchors", "rotation_origins", "rotation_anchors",
              "position_scales", "forearm_alignments")
    return {side: {name: json_safe(getattr(mapper, name).get(side)) for name in fields}
            for side in sides}


def matrix_quat(matrix):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).ravel())
    return quat


def software_info():
    from importlib import metadata

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                    text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {"mujoco": mujoco.__version__, "avp_stream": version("avp_stream"),
            "python": platform.python_version(), "git_commit": commit,
            "git_dirty": None if status is None else bool(status)}


class ContactTables:
    """Geom -> group lookups for the workspace scene, built once.

    Fails loudly if body/geom names change, rather than silently recording
    all-False contacts.
    """
    def __init__(self, model):
        n = model.ngeom
        body_names = [model.body(int(model.geom_bodyid[g])).name for g in range(n)]
        geom_names = [model.geom(g).name for g in range(n)]
        self.hand_group = {side: np.full(n, -1, dtype=np.int8) for side in SIDES}
        self.arm = {side: np.zeros(n, dtype=bool) for side in SIDES}
        prefixes = (("th", 0), ("ff", 1), ("mf", 2), ("rf", 3), ("lf", 4), ("palm", 5))
        for g, body in enumerate(body_names):
            for side in SIDES:
                hand = f"{side}_{side[0]}h_"
                if body.startswith(hand):
                    part = body[len(hand):]
                    self.hand_group[side][g] = next(
                        (group for prefix, group in prefixes if part.startswith(prefix)), 6)
                elif body in (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"):
                    self.arm[side][g] = True
        self.cup = np.array([body == "cup" for body in body_names])
        self.tray = np.array([body == "tray" for body in body_names])
        self.table = np.array([name == "worktop" or name.startswith("table_leg_")
                               for name in geom_names])
        self.limb = {side: (self.hand_group[side] >= 0) | self.arm[side] for side in SIDES}
        missing = [name for name, mask in (("cup", self.cup), ("tray", self.tray),
                                           ("worktop/table_leg_*", self.table))
                   if not mask.any()]
        for side in SIDES:
            if not self.arm[side].any():
                missing.append(f"{side}_shoulder/elbow/wrist")
            for index, group in enumerate(CONTACT_GROUPS):
                if not (self.hand_group[side] == index).any() and group != "other_hand":
                    missing.append(f"{side} hand {group}")
        if missing:
            raise ValueError("Workspace scene is missing expected geoms: " + ", ".join(missing))

    def measure(self, model, data):
        hand_cup = np.zeros((2, len(CONTACT_GROUPS)), dtype=bool)
        arm_cup = np.zeros(2, dtype=bool)
        force = np.zeros(2, dtype=np.float32)
        contacts = data.contact
        active = contacts.exclude == 0
        g1, g2 = contacts.geom1[active], contacts.geom2[active]
        index = np.flatnonzero(active)
        left, right = self.limb["left"], self.limb["right"]
        hand_hand = bool(np.any((left[g1] & right[g2]) | (left[g2] & right[g1])))
        cup_table = bool(np.any((self.cup[g1] & self.table[g2]) | (self.cup[g2] & self.table[g1])))
        cup_tray = bool(np.any((self.cup[g1] & self.tray[g2]) | (self.cup[g2] & self.tray[g1])))
        wrench = np.zeros(6)
        for i, a, b in zip(index, g1, g2):
            other = b if self.cup[a] else a if self.cup[b] else None
            if other is None:
                continue
            for s, side in enumerate(SIDES):
                group = self.hand_group[side][other]
                if group >= 0:
                    hand_cup[s, group] = True
                    mujoco.mj_contactForce(model, data, int(i), wrench)
                    force[s] += abs(wrench[0])
                arm_cup[s] |= bool(self.arm[side][other])
        return dict(hand_cup=hand_cup, arm_cup=arm_cup, hand_cup_force=force,
                    cup_table=cup_table, cup_tray=cup_tray, hand_hand=hand_hand,
                    ncon=np.int32(data.ncon))


# name -> (dtype, per-sample shape given model); used for buffers and checks.
def dataset_layout(model):
    k = len(FINGER_TARGET_NAMES)
    return {
        "time/sim": (np.float64, ()), "time/wall": (np.float64, ()),
        "robot/qpos": (np.float64, (model.nq,)), "robot/qvel": (np.float64, (model.nv,)),
        "robot/ctrl": (np.float64, (model.nu,)),
        "robot/wrist_pos": (np.float64, (2, 3)), "robot/wrist_quat": (np.float64, (2, 4)),
        "robot/target_pos": (np.float64, (2, 3)), "robot/target_quat": (np.float64, (2, 4)),
        "robot/reach_error": (np.float32, (2,)),
        "input/fresh": (bool, (2,)), "input/packet_age": (np.float32, (2,)),
        "input/packet_seq": (np.int64, (2,)), "input/new_packet": (bool, (2,)),
        "input/wrist": (np.float32, (2, 4, 4)),
        "input/finger_skeleton": (np.float32, (2, 25, 4, 4)),
        "input/arm_skeleton": (np.float32, (2, 27, 4, 4)),
        "input/forearm_vector": (np.float32, (2, 3)),
        "input/finger_targets": (np.float32, (2, k)),
        "scene/cup_pose": (np.float64, (7,)), "scene/cup_linvel": (np.float32, (3,)),
        "scene/cup_angvel": (np.float32, (3,)),
        "contacts/hand_cup": (bool, (2, len(CONTACT_GROUPS))), "contacts/arm_cup": (bool, (2,)),
        "contacts/hand_cup_force": (np.float32, (2,)), "contacts/cup_table": (bool, ()),
        "contacts/cup_tray": (bool, ()), "contacts/hand_hand": (bool, ()),
        "contacts/ncon": (np.int32, ()),
    }


def _array_or_nan(value, shape):
    try:
        array = np.asarray(value, dtype=float)
        if array.shape != shape:
            array = array.reshape(shape)
        return array
    except (ValueError, TypeError):
        return np.full(shape, np.nan)


class EpisodeRecorder:
    """Start/stop episodes of the human-arm workspace scene.

    Call sample() once per loop iteration in which physics advanced, never on
    paused/frozen iterations; watch() on every iteration while recording.
    """
    def __init__(self, model, controller, hands, *, record_dir=DEFAULT_RECORD_DIR,
                 task="cup_to_tray", operator="", source="keyboard", wrist_mode="human",
                 workspace=True, scale=0.6, video=True, video_fps=30, video_size=(640, 480),
                 max_seconds=120.0):
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not TASK_PATTERN.match(task):
            raise ValueError("task must be 1-40 letters, digits, '_' or '-'")
        if len(operator) > 32:
            raise ValueError("operator tag must be at most 32 characters")
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        if not workspace or wrist_mode != "human":
            raise ValueError("Recording requires --wrist human --workspace")
        self.model, self.controller, self.hands = model, controller, hands
        self.record_dir = Path(record_dir)
        self.task, self.operator, self.source = task, operator, source
        self.wrist_mode, self.workspace, self.scale = wrist_mode, workspace, scale
        self.video = video
        self.video_fps = video_fps
        self.video_size = tuple(video_size)
        self.max_seconds = float(max_seconds)
        self.contacts = ContactTables(model)
        self.layout = dataset_layout(model)
        self.cup_body = model.body("cup").id
        self.recording = False
        self.directory = None
        self.notice = None
        self.last_saved = None
        self.counter = None
        self.saved = []

    # -- lifecycle -------------------------------------------------------
    def _next_counter(self):
        used = [int(match.group(1)) for path in (self.record_dir.iterdir()
                if self.record_dir.is_dir() else [])
                if (match := EPISODE_PATTERN.match(path.name))]
        return max(used, default=0) + 1

    def start(self, data):
        if self.recording:
            raise RuntimeError("Already recording")
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.counter = self._next_counter()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.episode_id = f"{stamp}_{self.task}_{self.counter:04d}"
        self.directory = self.record_dir / self.episode_id
        self.directory.mkdir()
        self.buffers = {name: [] for name in self.layout}
        self.events = []
        self.time_base = None
        self.previous_seq = np.full(2, -1, dtype=np.int64)
        self.previous_fresh = None
        self.previous_reach = np.zeros(2, dtype=bool)
        self.start_sim, self.start_wall = data.time, time.monotonic()
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.recording = True
        self.notice = None
        self.event(data, "record_start", source=self.source)
        return self.directory

    def _times(self, data):
        base = self.time_base or (self.start_sim, self.start_wall)
        return max(0., data.time - base[0]), max(0., time.monotonic() - base[1])

    def event(self, data, kind, **payload):
        if not self.recording:
            return
        t, wall = self._times(data)
        payload = {key: scrub(value) if isinstance(value, str) else value
                   for key, value in payload.items()}
        self.events.append(json_safe({"t": t, "wall": wall, "sample": len(self.buffers["time/sim"]),
                                      "kind": kind, **payload}))

    def elapsed(self, data):
        return self._times(data)[0] if self.recording else 0.

    def expired(self, data):
        return self.recording and self.time_base is not None and \
            data.time - self.time_base[0] >= self.max_seconds

    # -- per-iteration hooks ----------------------------------------------
    def watch(self, data, fresh):
        """Log tracking and reach-limit edges; call on every loop iteration."""
        if not self.recording:
            return
        current = np.array([bool(fresh.get(side)) for side in SIDES])
        if self.source == "live":
            if self.previous_fresh is None:
                self.events[0]["fresh"] = dict(zip(SIDES, current.tolist()))
            else:
                for s, side in enumerate(SIDES):
                    if current[s] != self.previous_fresh[s]:
                        self.event(data, "tracking_regained" if current[s] else "tracking_lost", side=side)
            self.previous_fresh = current
        errors = getattr(self.controller, "reach_errors", {})
        reach = np.array([errors.get(side, 0.) > .005 for side in SIDES])
        for s, side in enumerate(SIDES):
            if reach[s] != self.previous_reach[s]:
                self.event(data, "reach_limit_on" if reach[s] else "reach_limit_off", side=side)
        self.previous_reach = reach

    def sample(self, data, *, samples, fresh, paused=False, mapper=None):
        """Append one sample. Only call after physics advanced (steps > 0).

        `mapper` is accepted for interface stability; calibration state is
        recorded through `calibrate` events instead of per sample.
        """
        if not self.recording or paused:
            return
        now = time.monotonic()
        if self.time_base is None:
            self.time_base = (data.time, now)
        row = {"time/sim": data.time - self.time_base[0], "time/wall": now - self.time_base[1]}
        model, controller = self.model, self.controller
        row["robot/qpos"] = data.qpos.copy()
        row["robot/qvel"] = data.qvel.copy()
        row["robot/ctrl"] = data.ctrl.copy()
        sites = [controller.sites[side] for side in SIDES]
        row["robot/wrist_pos"] = data.site_xpos[sites].copy()
        row["robot/wrist_quat"] = np.array([matrix_quat(data.site_xmat[site]) for site in sites])
        row["robot/target_pos"] = np.array([controller.targets[side] for side in SIDES])
        row["robot/target_quat"] = np.array([matrix_quat(controller.rotation_targets[side])
                                             for side in SIDES])
        errors = getattr(controller, "reach_errors", {})
        row["robot/reach_error"] = [errors.get(side, 0.) for side in SIDES]

        k = len(FINGER_TARGET_NAMES)
        inputs = {
            "input/fresh": np.zeros(2, bool), "input/packet_age": np.full(2, np.nan),
            "input/packet_seq": np.full(2, -1, np.int64), "input/new_packet": np.zeros(2, bool),
            "input/wrist": np.full((2, 4, 4), np.nan),
            "input/finger_skeleton": np.full((2, 25, 4, 4), np.nan),
            "input/arm_skeleton": np.full((2, 27, 4, 4), np.nan),
            "input/forearm_vector": np.full((2, 3), np.nan),
            "input/finger_targets": np.full((2, k), np.nan),
        }
        for s, side in enumerate(SIDES):
            inputs["input/fresh"][s] = bool(fresh.get(side))
            entry = samples.get(side)
            if entry is None:
                continue
            inputs["input/packet_age"][s] = now - entry[0]
            targets = entry[1]
            inputs["input/finger_targets"][s] = [targets.get(name, np.nan) for name in FINGER_TARGET_NAMES]
            if len(entry) > 2:
                inputs["input/wrist"][s] = _array_or_nan(entry[2], (4, 4))
            if len(entry) > 3 and entry[3] is not None:
                inputs["input/forearm_vector"][s] = _array_or_nan(entry[3], (3,))
            if len(entry) > 4 and entry[4] is not None:
                inputs["input/finger_skeleton"][s] = _array_or_nan(entry[4], (25, 4, 4))
            if len(entry) > 5 and entry[5] is not None:
                arm = np.asarray(entry[5], dtype=float)
                if arm.ndim == 3 and arm.shape[0] >= 27 and arm.shape[1:] == (4, 4):
                    inputs["input/arm_skeleton"][s] = arm[:27]
            if len(entry) > 6:
                inputs["input/packet_seq"][s] = entry[6]
        inputs["input/new_packet"] = (inputs["input/packet_seq"] >= 0) & \
            (inputs["input/packet_seq"] != self.previous_seq)
        self.previous_seq = inputs["input/packet_seq"].copy()
        row.update(inputs)

        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.cup_body, velocity, 0)
        row["scene/cup_pose"] = np.r_[data.xpos[self.cup_body], data.xquat[self.cup_body]]
        row["scene/cup_angvel"] = velocity[:3]
        row["scene/cup_linvel"] = velocity[3:]
        for name, value in self.contacts.measure(model, data).items():
            row["contacts/" + name] = value

        for name, value in row.items():
            self.buffers[name].append(np.asarray(value, dtype=self.layout[name][0]))

    def maybe_render(self, data):
        """Video frames are added in a later step; state-only for now."""

    # -- finishing ----------------------------------------------------------
    def metadata(self, end_reason, samples, duration):
        from workspace_scene import CUP_START, TABLE_TOP, TRAY_CENTER
        model = self.model
        return json_safe({
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id, "task": self.task, "source": self.source,
            "operator": self.operator, "created_utc": self.created,
            "duration_sim_s": round(float(duration), 6), "samples": samples,
            "end_reason": end_reason,
            "scene": {"wrist_mode": self.wrist_mode, "workspace": self.workspace,
                      "scale": self.scale, "cup_start": CUP_START,
                      "tray_center": TRAY_CENTER, "table_top": TABLE_TOP},
            "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                      "timestep": model.opt.timestep,
                      "joint_names": [model.joint(i).name for i in range(model.njnt)],
                      "actuator_names": [model.actuator(i).name for i in range(model.nu)]},
            "software": software_info(),
            "finger_target_names": FINGER_TARGET_NAMES,
            "video": {"status": "disabled"},
            "labels": None,
        })

    def stop(self, data, end_reason):
        """Write episode.h5 + episode.json; return the episode directory.

        Returns None (and deletes the folder) if no samples were recorded,
        for example when E was pressed twice while paused.
        """
        if not self.recording:
            return None
        if end_reason not in END_REASONS:
            raise ValueError(f"end_reason must be one of {END_REASONS}")
        self.event(data, "record_stop", end_reason=end_reason)
        self.recording = False
        directory = self.directory
        count = len(self.buffers["time/sim"])
        if not count:
            shutil.rmtree(directory, ignore_errors=True)
            self.notice = f"Episode {self.counter:04d} had no samples (paused?); nothing saved"
            print(self.notice, flush=True)
            return None
        arrays = {name: np.stack(values) for name, values in self.buffers.items()}
        self.buffers = None
        metadata = self.metadata(end_reason, count, arrays["time/sim"][-1])
        self._write(directory, arrays, metadata)
        self.last_saved = (self.counter, metadata)
        self.saved.append(directory)
        self.notice = None
        print(f"Saved episode: {relative_path(directory)} ({count} samples, "
              f"{metadata['duration_sim_s']:.1f} s sim, end: {end_reason})", flush=True)
        return directory

    def _write(self, directory, arrays, metadata):
        import h5py
        partial = directory / "episode.h5.partial"
        with h5py.File(partial, "w") as file:
            file.attrs["schema"] = SCHEMA
            file.attrs["schema_version"] = SCHEMA_VERSION
            file.attrs["metadata"] = json.dumps(metadata)
            file.attrs["events"] = json.dumps(self.events)
            for name, values in arrays.items():
                dataset = file.create_dataset(name, data=values, compression="gzip")
                if name == "input/finger_targets":
                    dataset.attrs["names"] = json.dumps(list(FINGER_TARGET_NAMES))
                elif name == "contacts/hand_cup":
                    dataset.attrs["groups"] = json.dumps(list(CONTACT_GROUPS))
        os.replace(partial, directory / "episode.h5")
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def discard(self):
        """Stop and delete the current episode; nothing is kept."""
        if not self.recording:
            return
        self.recording = False
        self.buffers = None
        shutil.rmtree(self.directory, ignore_errors=True)
        self.notice = f"Discarded ep {self.counter:04d}"
        print(self.notice, flush=True)

    def status_text(self, data):
        if self.recording:
            seconds = self.elapsed(data)
            text = f"● REC {int(seconds // 60):02d}:{seconds % 60:04.1f} · ep {self.counter:04d}"
            return text + (f" · {self.notice}" if self.notice else "")
        if self.notice:
            return self.notice
        if self.last_saved:
            counter, metadata = self.last_saved
            labels = metadata.get("labels") or {}
            outcome = labels.get("outcome") or {}
            if outcome.get("success") is True:
                result = "SUCCESS"
            elif outcome.get("success") is False:
                result = f"FAIL ({outcome.get('failure_reason')})"
            else:
                result = "unlabeled"
            return f"Saved ep {counter:04d} · {result}"
        return ""
