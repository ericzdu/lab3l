"""Read-only episode access for lab3l Studio (demo cut): list, detail, video path.

Every folder name from a request goes through `resolve()`, which only returns
real episode folders directly inside the episodes directory (not `_junk`, no
`..`, no absolute paths, no symlinks pointing elsewhere).
"""
from __future__ import annotations

import json
from pathlib import Path
import re

CAMERAS = ("overview", "operator")
FOLDER_PATTERN = re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9_-]+_(\d{4,})$")
# Projector wording for each labeler failure_reason (episode_labels.FAILURE_ORDER).
FRIENDLY_REASONS = {
    "never_touched": "never touched the cup",
    "never_lifted": "never lifted",
    "dropped_outside_tray": "dropped outside tray",
    "tipped_over": "cup tipped over",
    "not_in_tray": "not in the tray",
    "not_released": "never let go",
    "not_settled": "cup still moving",
    "too_short": "too short",
}
PLACEMENT = {"placed": "placed", "dropped_in": "dropped in"}
# Phases whose duration the one-line strip shows; the rest are named only.
TIMED_PHASES = ("reach", "grasp", "transport", "place")


def friendly_reason(reason):
    if reason is None:
        return None
    return FRIENDLY_REASONS.get(reason, str(reason).replace("_", " "))


def resolve(root, folder):
    """Episode folder Path for a request's folder name, or None if not allowed."""
    if not isinstance(folder, str) or not FOLDER_PATTERN.match(folder):
        return None
    root = Path(root).resolve()
    path = root / folder
    try:
        real = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if real.parent != root or not real.is_dir() or not (real / "episode.json").is_file():
        return None
    return real


def read_metadata(path):
    return json.loads((Path(path) / "episode.json").read_text())


def outcome_of(metadata):
    return ((metadata.get("labels") or {}).get("outcome")) or {}


def summary(path, metadata):
    outcome = outcome_of(metadata)
    match = FOLDER_PATTERN.match(path.name)
    success = outcome.get("success")
    return {
        "folder": path.name,
        "number": int(match.group(1)),
        "created_utc": metadata.get("created_utc"),
        "duration_s": round(float(metadata.get("duration_sim_s") or 0), 1),
        "outcome": "success" if success is True else "fail" if success is False else "unlabeled",
        "failure_reason": outcome.get("failure_reason"),
        "friendly_reason": friendly_reason(outcome.get("failure_reason")),
        "confidence": outcome.get("confidence"),
        "placement": PLACEMENT.get(outcome.get("placement")),
        "grasp_assist": bool(metadata.get("grasp_assist")),
        "source": metadata.get("source"),
    }


def list_episodes(root):
    """Kept episodes (not `_junk`), newest first, plus header counts."""
    root = Path(root)
    rows = []
    if root.is_dir():
        for path in root.iterdir():
            real = resolve(root, path.name)
            if real is None:
                continue
            try:
                rows.append(summary(real, read_metadata(real)))
            except (OSError, ValueError):
                continue
    rows.sort(key=lambda row: (row["created_utc"] or "", row["number"]), reverse=True)
    return {"episodes": rows, "count": len(rows),
            "successes": sum(row["outcome"] == "success" for row in rows)}


def phase_strip(segments):
    """'reach 1.1 s · grasp 1.1 s · lift · transport 3.8 s · ...' (idle/recover left out)."""
    parts, previous = [], None
    durations = {}
    order = []
    for segment in segments or []:
        phase = segment.get("phase")
        if phase in (None, "idle"):
            continue
        if phase != previous:
            order.append(phase)
            durations[len(order) - 1] = 0.
        durations[len(order) - 1] += float(segment["end_t"]) - float(segment["start_t"])
        previous = phase
    for index, phase in enumerate(order):
        parts.append(f"{phase} {durations[index]:.1f} s" if phase in TIMED_PHASES else phase)
    return " · ".join(parts)


def detail(path):
    """Everything the result card needs: outcome, segments, events and video timing."""
    import h5py
    metadata = read_metadata(path)
    labels = metadata.get("labels") or {}
    video = metadata.get("video") or {}
    cameras = {}
    with h5py.File(Path(path) / "episode.h5", "r") as file:
        duration = float(file["time/sim"][-1]) if len(file["time/sim"]) else 0.
        for camera in CAMERAS:
            name = f"video/{camera}/frame_time"
            if (Path(path) / f"cam_{camera}.mp4").is_file() and name in file:
                cameras[camera] = {"frame_time": [round(float(t), 4) for t in file[name][...]]}
    return {
        **summary(path, metadata),
        "duration_s": round(duration, 3),
        "labeler_version": labels.get("labeler_version"),
        "segments": [{key: segment.get(key) for key in ("phase", "side", "start_t", "end_t", "confidence")}
                     for segment in labels.get("segments") or []],
        "events": [{key: event.get(key) for key in ("t", "kind", "side")}
                   for event in labels.get("events") or []],
        "phase_strip": phase_strip(labels.get("segments")),
        "video_fps": video.get("fps"),
        "cameras": cameras,
    }


def video_path(root, folder, camera):
    path = resolve(root, folder)
    if path is None or camera not in CAMERAS:
        return None
    mp4 = path / f"cam_{camera}.mp4"
    return mp4 if mp4.is_file() and not mp4.is_symlink() else None
