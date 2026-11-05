"""Heuristic success and phase labels for recorded cup-to-tray episodes.

label_episode() is a pure function of the recorded arrays and metadata: no
MuJoCo stepping, so it is fast, testable with synthetic data and re-runnable
on old episodes when the thresholds below are recalibrated. The thresholds
and confidence scores are hand-chosen heuristics, not learned or validated
against human labels yet; every factor is stored so the team can retune it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import workspace_scene
from workspace_scene import CUP_HALF_HEIGHT, TRAY_HALF_EXTENTS, TRAY_RIM_THICKNESS


LABELER_VERSION = "2"
PHASES = ("idle", "reach", "grasp", "lift", "transport", "place", "release", "retreat", "recover")
IDLE, REACH, GRASP, LIFT, TRANSPORT, PLACE, RELEASE, RETREAT, RECOVER = range(len(PHASES))
SIDES = ("left", "right")
FAILURE_ORDER = ("never_touched", "never_lifted", "dropped_outside_tray", "tipped_over",
                 "not_in_tray", "not_released", "not_settled", "too_short")
BOUNDARY_WINDOW = .2   # s; flicker search around a segment boundary
MARGIN_WINDOW = .1     # s; margin averaging after a threshold boundary


@dataclass(frozen=True)
class LabelConfig:
    """Every threshold the labeler uses; saved with the labels."""
    lift_height: float = 0.010        # m above the cup's resting height on the table
    settle_window: float = 0.5        # s at end of episode
    settle_lin_speed: float = 0.02    # m/s
    settle_ang_speed: float = 0.5     # rad/s
    upright_max_deg: float = 25.0
    debounce: float = 0.10            # s; minimum duration for a contact/support state change
    drop_min_fall: float = 0.02       # m of descent while unsupported & untouched
    reach_speed: float = 0.05         # m/s wrist speed toward cup
    transport_speed: float = 0.05     # m/s horizontal cup speed
    transport_distance: float = 0.03  # m horizontal displacement from lift-off
    retreat_distance: float = 0.03    # m wrist moves away from cup after release
    retreat_time: float = 0.3         # s after release
    tip_over_deg: float = 60.0
    bimanual_min: float = 0.3         # s of simultaneous two-hand opposition
    min_duration: float = 1.0         # s; shorter episodes are too_short
    drop_in_window: float = 1.0       # s; let go in the air, cup lands in the tray within this


# -- loading / writing ----------------------------------------------------------
NEEDED = ("time/sim", "robot/wrist_pos", "input/fresh", "scene/cup_pose", "scene/cup_linvel",
          "scene/cup_angvel", "contacts/hand_cup", "contacts/arm_cup", "contacts/cup_table",
          "contacts/cup_tray", "contacts/hand_hand")


def scene_geometry(metadata):
    """Prop positions an episode was recorded with, from metadata.scene.

    Each of table_top / cup_start / tray_center falls back only when missing:
    to the version-1 layout if the episode has no scene_version (recorded
    before it existed), else to the current workspace_scene constants. Cup
    and tray sizes have not changed between scene versions.
    """
    scene = (metadata or {}).get("scene") or {}
    version = int(scene.get("scene_version", 1))
    if version == 1:
        defaults = workspace_scene.SCENE_V1
    else:
        defaults = {"table_top": workspace_scene.TABLE_TOP,
                    "cup_start": workspace_scene.CUP_START,
                    "tray_center": workspace_scene.TRAY_CENTER}
    value = lambda key: scene[key] if scene.get(key) is not None else defaults[key]
    return {"scene_version": version, "table_top": float(value("table_top")),
            "cup_start": np.asarray(value("cup_start"), dtype=float),
            "tray_center": np.asarray(value("tray_center"), dtype=float)[:2]}


def episode_path(path):
    path = Path(path)
    return path / "episode.h5" if path.is_dir() else path


def load_episode(path):
    """(arrays, metadata) for label_episode from an episode.h5 or its folder."""
    import h5py
    with h5py.File(episode_path(path), "r") as file:
        if file.attrs.get("schema") != "lab3l.episode":
            raise ValueError(f"{path} is not a lab3l.episode file")
        arrays = {name: file[name][...] for name in NEEDED}
        metadata = json.loads(file.attrs["metadata"])
    return arrays, metadata


def labels_json(labels):
    """JSON-safe part of the labels stored in episode.json / metadata."""
    return {key: labels[key] for key in ("labeler_version", "created_utc", "config", "outcome",
                                         "segments", "events")}


def write_labels(path, labels):
    """Replace labels/ in episode.h5 and labels in its metadata + episode.json.

    created_utc is kept when the labels are otherwise unchanged, so labeling
    twice is idempotent.
    """
    import h5py
    path = episode_path(path)
    with h5py.File(path, "a") as file:
        metadata = json.loads(file.attrs["metadata"])
        summary = labels_json(labels)
        previous = metadata.get("labels")
        if previous and {**previous, "created_utc": None} == {**json.loads(json.dumps(summary)),
                                                              "created_utc": None}:
            summary["created_utc"] = labels["created_utc"] = previous["created_utc"]
        if "labels" in file:
            del file["labels"]
        group = file.create_group("labels")
        phase = group.create_dataset("phase", data=labels["phase"].astype(np.int8), compression="gzip")
        phase.attrs["names"] = json.dumps(list(PHASES))
        for name in ("segments", "events", "outcome", "config"):
            group.attrs[name] = json.dumps(labels[name])
        group.attrs["labeler_version"] = labels["labeler_version"]
        group.attrs["created_utc"] = labels["created_utc"]
        metadata["labels"] = summary
        file.attrs["metadata"] = json.dumps(metadata)
    path.with_name("episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


# -- signal helpers -----------------------------------------------------------------
def debounce(raw, t, window):
    """Accept a state change only if the new value holds for >= window seconds.

    Accepted changes keep their raw onset time; shorter runs are absorbed
    into the surrounding state. The first sample's value is the initial state.
    """
    raw = np.asarray(raw, dtype=bool)
    out = np.empty_like(raw)
    if not len(raw):
        return out
    starts = np.r_[0, np.flatnonzero(raw[1:] != raw[:-1]) + 1]
    step = t[-1] - t[-2] if len(t) > 1 else 0.
    ends_t = np.r_[t[starts[1:]], t[-1] + step]
    state = raw[0]
    for start, end, end_t in zip(starts, np.r_[starts[1:], len(raw)], ends_t):
        if raw[start] != state and end_t - t[start] >= window - 1e-9:
            state = raw[start]
        out[start:end] = state
    return out


def rising(signal):
    signal = np.asarray(signal, dtype=bool)
    return np.flatnonzero(signal & ~np.r_[False, signal[:-1]])


def falling(signal):
    signal = np.asarray(signal, dtype=bool)
    return np.flatnonzero(~signal & np.r_[False, signal[:-1]])


def tilt_degrees(quat):
    """Angle between the cup's local +z and world +z (wxyz quaternions)."""
    x, y = quat[:, 1], quat[:, 2]
    norm = np.sum(quat ** 2, axis=1)
    return np.rad2deg(np.arccos(np.clip(1 - 2 * (x * x + y * y) / np.maximum(norm, 1e-12), -1, 1)))


def ratio_margin(signal, threshold):
    return np.minimum(1., np.abs(np.asarray(signal, dtype=float) - threshold) / threshold)


class Signals:
    """Raw and debounced per-sample signals derived from recorded arrays."""
    def __init__(self, arrays, metadata, cfg):
        self.cfg = cfg
        t = self.t = np.asarray(arrays["time/sim"], dtype=float)
        self.n = len(t)
        hand_cup = np.asarray(arrays["contacts/hand_cup"], dtype=bool)
        arm_cup = np.asarray(arrays["contacts/arm_cup"], dtype=bool)
        d = lambda raw: debounce(raw, t, cfg.debounce)
        self.touch_raw = hand_cup.any(axis=2) | arm_cup
        self.touch = np.column_stack([d(self.touch_raw[:, s]) for s in range(2)])
        self.opposition_raw = hand_cup[:, :, 0] & hand_cup[:, :, 1:6].any(axis=2)
        self.opposition = np.column_stack([d(self.opposition_raw[:, s]) for s in range(2)])
        self.table_raw = np.asarray(arrays["contacts/cup_table"], dtype=bool)
        self.tray_raw = np.asarray(arrays["contacts/cup_tray"], dtype=bool)
        self.table, self.tray = d(self.table_raw), d(self.tray_raw)
        self.supported_raw = self.table_raw | self.tray_raw
        self.supported = d(self.supported_raw)
        pose = np.asarray(arrays["scene/cup_pose"], dtype=float)
        self.cup = pose[:, :3]
        scene = scene_geometry(metadata)
        rest = scene["table_top"] + CUP_HALF_HEIGHT
        self.height = self.cup[:, 2] - rest
        self.lifted_raw = ~self.supported_raw & (self.height > cfg.lift_height)
        self.lifted = d(self.lifted_raw)
        self.tilt = tilt_degrees(pose[:, 3:7])
        self.tipped = d(self.tilt > cfg.tip_over_deg)
        self.lin_speed = np.linalg.norm(np.asarray(arrays["scene/cup_linvel"], dtype=float), axis=1)
        self.ang_speed = np.linalg.norm(np.asarray(arrays["scene/cup_angvel"], dtype=float), axis=1)
        self.h_speed = np.linalg.norm(np.asarray(arrays["scene/cup_linvel"], dtype=float)[:, :2], axis=1)
        self.hand_hand = d(np.asarray(arrays["contacts/hand_hand"], dtype=bool))
        self.wrist = np.asarray(arrays["robot/wrist_pos"], dtype=float)
        self.fresh = np.asarray(arrays["input/fresh"], dtype=bool)
        self.live = metadata.get("source") == "live"
        self.tray_offset = np.abs(self.cup[:, :2] - scene["tray_center"])
        self.tray_bounds = np.asarray(TRAY_HALF_EXTENTS) - TRAY_RIM_THICKNESS

    def window(self, start, seconds):
        """Sample indices in [t[start], t[start] + seconds]."""
        end = np.searchsorted(self.t, self.t[start] + seconds + 1e-9, side="right")
        return np.arange(start, max(start + 1, end))

    def fresh_fraction(self, indices, side):
        if not self.live or not len(indices):
            return 1.
        return float(self.fresh[indices, side].mean())


def free_fall_drops(sig):
    """Drop events: unsupported and untouched >= debounce while falling >= drop_min_fall."""
    cfg = sig.cfg
    free = ~sig.supported_raw & ~sig.touch_raw.any(axis=1)
    drops = []
    for start in rising(free):
        end = start
        while end < sig.n and free[end]:
            end += 1
        if sig.t[end - 1] - sig.t[start] < cfg.debounce - 1e-9 and end < sig.n:
            continue
        z = sig.cup[start:end, 2]
        fall = float(np.max(np.maximum.accumulate(z) - z)) if len(z) else 0.
        if fall < cfg.drop_min_fall:
            continue
        landing = "none"
        if end < sig.n:
            landing = "tray" if sig.tray_raw[end] else "table" if sig.table_raw[end] else "other"
        drops.append(dict(sample=int(start), fall=fall, landed=landing))
    return drops


# -- phases ---------------------------------------------------------------------------
def lands_in_tray(sig, active, i):
    """After the active hand lets go at sample i, does the cup land in the tray untouched?

    Looks ahead up to drop_in_window seconds for the first support contact
    or re-touch. A drop-in placement is a release, not a drop or a failure.
    Returns (landing sample, fall in metres) or None.
    """
    cfg = sig.cfg
    stop = np.searchsorted(sig.t, sig.t[i] + cfg.drop_in_window + 1e-9, side="right")
    for j in range(i, min(stop, sig.n)):
        if sig.touch_raw[j, active]:
            return None
        if sig.supported_raw[j]:
            if sig.tray_raw[j] and not sig.table_raw[j]:
                return j, float(max(0., sig.cup[i, 2] - sig.cup[j, 2]))
            return None
    return None


def segment_phases(sig, active, drops):
    """Per-sample phase codes and boundary causes from the state machine.

    Labeler v2: a grasp needs thumb + finger opposition (or the cup actually
    lifting); a touch without it is still reaching (fumbling). Letting go
    above the tray and the cup landing in it is a release ("drop_in"). After
    a release, touches without opposition are adjustments, not new grasps.
    Every re-entry into grasp after the first counts as a regrasp.
    """
    cfg, n, t = sig.cfg, sig.n, sig.t
    a = active
    phase = np.zeros(n, dtype=np.int8)
    transitions = []  # (sample, new phase, cause)
    drop_at = {d["sample"]: d for d in drops}
    toward = np.zeros(n)
    if n > 1:
        velocity = np.gradient(sig.wrist[:, a], t, axis=0)
        offset = sig.cup - sig.wrist[:, a]
        direction = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-9)
        toward = np.sum(velocity * direction, axis=1)
    sig.toward = toward
    reaching = debounce(toward > cfg.reach_speed, t, cfg.debounce)
    touch = sig.touch[:, a]
    opposition = sig.opposition[:, a]
    opposition_onsets = set(rising(opposition).tolist())
    lifted_seen = False
    state, lift_xy, release_index, release_distance = IDLE, None, None, None
    drop_ins = []  # (release sample, landing sample, fall)
    grasp_entries = 0

    def go(i, new, cause):
        nonlocal state, grasp_entries
        if new != state:
            if new == GRASP:
                grasp_entries += 1
            transitions.append((i, new, cause))
            state = new

    for i in range(n):
        distance = np.linalg.norm(sig.cup[i] - sig.wrist[i, a])
        table_again = lifted_seen and sig.table[i] and not sig.table[i - 1]
        lifted_seen |= bool(sig.lifted[i])
        landing = (lands_in_tray(sig, a, i)
                   if state in (LIFT, TRANSPORT) and not touch[i] else None)
        if landing is not None:
            drop_ins.append((i, *landing))
            go(i, RELEASE, "drop_in")
        elif (i in drop_at and state != RECOVER
              and not (drop_at[i]["landed"] == "tray" and state in (RELEASE, RETREAT))):
            go(i, RECOVER, "drop")
        elif table_again and state != RECOVER:
            go(i, RECOVER, "table_contact")
        elif state == IDLE:
            if opposition[i] or (touch[i] and sig.lifted[i]):
                go(i, GRASP, "opposition")
            elif touch[i]:
                go(i, REACH, "touch")
            elif reaching[i]:
                go(i, REACH, "reach_speed")
        elif state == REACH:
            if opposition[i] or (touch[i] and sig.lifted[i]):
                go(i, GRASP, "opposition")
        elif state == GRASP:
            if sig.lifted[i]:
                go(i, LIFT, "lifted")
            elif not touch[i]:
                go(i, REACH, "touch")
        elif state == LIFT:
            moved = np.linalg.norm(sig.cup[i, :2] - lift_xy) > cfg.transport_distance
            if sig.tray[i] and touch[i]:
                go(i, PLACE, "tray_contact")
            elif sig.h_speed[i] > cfg.transport_speed or moved:
                go(i, TRANSPORT, "transport_distance" if moved else "transport_speed")
        elif state == TRANSPORT:
            if sig.tray[i] and touch[i]:
                go(i, PLACE, "tray_contact")
        elif state == PLACE:
            if not touch[i] and sig.tray[i]:
                go(i, RELEASE, "touch")
            elif sig.lifted[i]:
                go(i, LIFT, "lifted")
        elif state in (RELEASE, RETREAT):
            if i in opposition_onsets or (touch[i] and sig.lifted[i]):
                go(i, GRASP, "opposition")
            elif state == RELEASE and distance - release_distance > cfg.retreat_distance:
                go(i, RETREAT, "retreat_distance")
            elif state == RELEASE and t[i] - t[release_index] >= cfg.retreat_time:
                go(i, RETREAT, "retreat_time")
        elif state == RECOVER:
            if touch[i] and sig.lifted[i]:
                go(i, LIFT, "lifted")
            elif i in opposition_onsets:
                go(i, GRASP, "opposition")
        if transitions and transitions[-1][0] == i:
            if state == LIFT:
                lift_xy = sig.cup[i, :2].copy()
            elif state == RELEASE:
                release_index, release_distance = i, distance
        phase[i] = state
    sig.drop_ins = drop_ins
    return phase, transitions, max(0, grasp_entries - 1)


CONTACT_CAUSES = {"touch": "touch", "tray_contact": "tray", "table_contact": "table",
                  "opposition": "opposition", "drop_in": "touch"}


def boundary_quality(sig, active, index, cause):
    """(flickers, margin) for a phase boundary caused by `cause` at `index`."""
    cfg, t = sig.cfg, sig.t
    if cause is None:
        return 0, 1.
    raw = {"touch": sig.touch_raw[:, active], "tray": sig.tray_raw, "table": sig.table_raw,
           "opposition": sig.opposition_raw[:, active],
           "lifted": sig.lifted_raw, "drop": ~sig.supported_raw & ~sig.touch_raw.any(axis=1),
           "reach_speed": sig.toward > cfg.reach_speed,
           "transport_speed": sig.h_speed > cfg.transport_speed}.get(CONTACT_CAUSES.get(cause, cause))
    flickers = 0
    if raw is not None:
        near = np.flatnonzero(np.abs(t - t[index]) <= BOUNDARY_WINDOW + 1e-9)
        toggles = int(np.count_nonzero(raw[near[1:]] != raw[near[:-1]])) if len(near) > 1 else 0
        flickers = max(0, toggles - 1)
    after = sig.window(index, MARGIN_WINDOW)
    margin = 1.
    if cause == "reach_speed":
        margin = ratio_margin(sig.toward[after], cfg.reach_speed).mean()
    elif cause == "lifted":
        margin = ratio_margin(sig.height[after], cfg.lift_height).mean()
    elif cause == "transport_speed":
        margin = ratio_margin(sig.h_speed[after], cfg.transport_speed).mean()
    elif cause == "drop":
        z = sig.cup[index:, 2]
        free = ~sig.supported_raw[index:]
        z = z[:np.argmin(free)] if not free.all() else z
        fall = float(np.max(np.maximum.accumulate(z) - z)) if len(z) else 0.
        margin = ratio_margin(fall, cfg.drop_min_fall)
    return flickers, float(margin)


def build_segments(sig, phase, transitions, active):
    starts = [0] + [i for i, _, _ in transitions if i > 0]
    causes = {i: cause for i, _, cause in transitions}
    segments = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else sig.n
        in_flicker, in_margin = boundary_quality(sig, active, start, causes.get(start))
        out_flicker = boundary_quality(sig, active, end, causes.get(end))[0] if end < sig.n else 0
        fresh = sig.fresh_fraction(np.arange(start, end), active)
        # Stability looks at both boundaries; margin at the one that opened
        # the segment (the 0.1 s after it lies inside the segment).
        stability = 1 / (1 + in_flicker + out_flicker)
        margin = in_margin
        code = int(phase[start])
        segments.append({
            "phase": PHASES[code], "side": None if code == IDLE else SIDES[active],
            "start_t": float(sig.t[start]), "end_t": float(sig.t[end] if end < sig.n else sig.t[-1]),
            "start_sample": int(start), "end_sample": int(end),
            "confidence": float(np.clip(fresh * stability * margin, 0, 1)),
            "fresh_frac": float(fresh), "stability": float(stability), "margin": float(margin),
            "start_cause": causes.get(start)})
    return segments


def label_events(sig, transitions, drops, active, phase):
    events = []

    def add(index, kind, side=None, **extra):
        events.append({"t": float(sig.t[index]), "sample": int(index), "kind": kind,
                       "side": side, **extra})

    any_touch = sig.touch.any(axis=1)
    if any_touch.any():
        index = int(np.argmax(any_touch))
        side = active if sig.touch[index, active] else 1 - active
        add(index, "first_touch", SIDES[side])
    for s, side in enumerate(SIDES):
        for i in rising(sig.opposition[:, s]):
            add(i, "grasp_opposition_start", side)
        for i in falling(sig.opposition[:, s]):
            add(i, "grasp_opposition_end", side)
    for i in rising(sig.lifted):
        add(i, "lift_off", SIDES[active])
    for i in rising(sig.tray):
        add(i, "tray_contact")
    for drop in drops:
        add(drop["sample"], "drop", None, fall=drop["fall"], landed=drop["landed"])
    for i in rising(sig.tipped):
        add(i, "tip_over")
    for i in rising(sig.hand_hand):
        add(i, "hand_hand_contact")
    grasps = 0
    for i, new, cause in transitions:
        if new == RELEASE:
            add(i, "release", SIDES[active], how="drop_in" if cause == "drop_in" else "placed")
        elif new == GRASP:
            grasps += 1
            if grasps > 1:
                add(i, "regrasp", SIDES[active])
    for release, landing, fall in getattr(sig, "drop_ins", []):
        add(landing, "drop_in", SIDES[active], fall=fall, released_at=float(sig.t[release]))
    after_release = np.isin(phase, (RELEASE, RETREAT))
    for i in rising(sig.touch[:, active]):
        if after_release[i] and not sig.opposition[i, active]:
            add(i, "post_release_touch", SIDES[active])
    events.sort(key=lambda event: (event["sample"], event["kind"]))
    return events


# -- outcome ------------------------------------------------------------------------
def cup_to_tray_outcome(sig, active, drops, segments):
    cfg = sig.cfg
    duration = float(sig.t[-1] - sig.t[0]) if sig.n else 0.
    window = np.flatnonzero(sig.t >= sig.t[-1] - cfg.settle_window - 1e-9) if sig.n else np.array([], int)
    inside = np.all(sig.tray_offset[window] <= sig.tray_bounds, axis=1)
    criteria = {
        "in_tray": bool(len(window) and inside.all() and sig.tray[window].all()
                        and not sig.table[window].any()),
        "upright": bool(len(window) and sig.tilt[window].max() < cfg.upright_max_deg),
        "released": bool(len(window) and not sig.touch[window].any()),
        "settled": bool(len(window) and sig.lin_speed[window].max() < cfg.settle_lin_speed
                        and sig.ang_speed[window].max() < cfg.settle_ang_speed),
        "lifted_at_least_once": bool(sig.lifted.any()),
    }
    long_enough = duration >= max(cfg.min_duration, cfg.settle_window)
    margins = {"released": 1.}
    if len(window):
        bounds = np.asarray(sig.tray_bounds)
        margins["in_tray"] = float(np.min(np.abs(bounds - sig.tray_offset[window]) / bounds).clip(0, 1))
        margins["upright"] = float(ratio_margin(sig.tilt[window].max(), cfg.upright_max_deg))
        margins["settled"] = float(min(ratio_margin(sig.lin_speed[window].max(), cfg.settle_lin_speed),
                                       ratio_margin(sig.ang_speed[window].max(), cfg.settle_ang_speed)))
    margins["lifted_at_least_once"] = float(ratio_margin(np.max(sig.height, initial=0.), cfg.lift_height))
    success = all(criteria.values()) and long_enough
    reason = None
    if not success:
        dropped_outside = any(drop["landed"] != "tray" for drop in drops)
        failing = {
            "never_touched": not sig.touch.any(),
            "never_lifted": not criteria["lifted_at_least_once"],
            "dropped_outside_tray": dropped_outside and not criteria["in_tray"],
            "tipped_over": not criteria["upright"],
            "not_in_tray": not criteria["in_tray"],
            "not_released": not criteria["released"],
            "not_settled": not criteria["settled"],
            "too_short": not long_enough,
        }
        reason = next(name for name in FAILURE_ORDER if failing[name])
    fresh = sig.fresh_fraction(window, active)
    regrasps = max(0, sum(1 for segment in segments if segment["phase"] == "grasp") - 1)
    releases = [segment for segment in segments if segment["phase"] == "release"]
    placement = None
    if releases:
        placement = "dropped_in" if releases[-1].get("start_cause") == "drop_in" else "placed"
    both = sig.opposition.all(axis=1)
    longest = 0.
    for start in rising(both):
        end = start
        while end < sig.n and both[end]:
            end += 1
        longest = max(longest, float(sig.t[min(end, sig.n - 1)] - sig.t[start]))
    return {
        "outcome": "success" if success else "failure", "success": success,
        "failure_reason": reason, "criteria": criteria, "margins": margins,
        "confidence": float(np.clip(min(margins.values()) * fresh, 0, 1)),
        "settle_fresh_frac": float(fresh), "duration_s": duration,
        "active_side": SIDES[active], "bimanual": longest > cfg.bimanual_min,
        "regrasp_count": regrasps, "placement": placement,
    }


def label_episode(arrays, metadata, cfg=LabelConfig()):
    """Phase labels, events and outcome for one episode (pure; no file I/O)."""
    sig = Signals(arrays, metadata, cfg)
    if sig.n < 2:
        raise ValueError("Episode needs at least two samples to label")
    contact_time = sig.touch.sum(axis=0)
    active = 0 if contact_time[0] > contact_time[1] else 1  # ties -> right
    drops = free_fall_drops(sig)
    phase, transitions, _ = segment_phases(sig, active, drops)
    segments = build_segments(sig, phase, transitions, active)
    events = label_events(sig, transitions, drops, active, phase)
    if metadata.get("task") == "cup_to_tray":
        outcome = cup_to_tray_outcome(sig, active, drops, segments)
    else:
        outcome = {"outcome": "not_evaluated", "success": None, "failure_reason": None,
                   "criteria": {}, "confidence": None, "active_side": SIDES[active],
                   "bimanual": None,
                   "regrasp_count": sum(1 for e in events if e["kind"] == "regrasp")}
    return {
        "labeler_version": LABELER_VERSION, "config": asdict(cfg),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": phase, "phase_names": list(PHASES), "segments": segments,
        "events": events, "outcome": outcome,
    }


def summary_line(episode_id, labels):
    outcome = labels["outcome"]
    if outcome.get("success") is None:
        result = "NOT EVALUATED"
    else:
        result = "SUCCESS" if outcome["success"] else f"FAIL ({outcome['failure_reason']})"
        result += f" conf {outcome['confidence']:.2f}"
    phases = []
    for segment in labels["segments"]:
        if not phases or phases[-1] != segment["phase"]:
            phases.append(segment["phase"])
    return (f"{episode_id}: {result} · {outcome['active_side']} hand · "
            f"regrasps {outcome['regrasp_count']} · " + " > ".join(phases))
