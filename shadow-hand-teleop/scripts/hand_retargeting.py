"""Conservative, uncalibrated finger-flexion mapping for the first live demo.

Uses VisionProTeleop's 25-joint wrist-local skeleton. This is not full-pose IK:
finger spread, thumb opposition and wrist/arm motion are not reconstructed.
"""

from __future__ import annotations

import numpy as np


def bend(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-8:
        raise ValueError("Missing or degenerate finger tracking")
    return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1)))


def finger_targets(joints: np.ndarray) -> dict[str, float]:
    """Return named Shadow actuator targets, invariant to rigid pose and scale."""
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (25, 4, 4) or not np.isfinite(joints).all():
        raise ValueError("Expected 25 finite joint transforms")
    positions = joints[:, :3, 3]
    targets = {"WRJ1": 0.0, "WRJ2": 0.0, "LFJ5": 0.0}
    for start, prefix in ((5, "FF"), (10, "MF"), (15, "RF"), (20, "LF")):
        bones = np.diff(positions[start : start + 5], axis=0)
        # MCP flexion drives J3; Shadow's tendon J0 drives PIP + DIP together.
        targets[f"{prefix}J4"] = 0.0
        targets[f"{prefix}J3"] = bend(bones[0], bones[1])
        targets[f"{prefix}J0"] = bend(bones[1], bones[2]) + bend(bones[2], bones[3])
    thumb = np.diff(positions[1:5], axis=0)
    targets.update(THJ5=0.0, THJ4=0.0, THJ3=0.0,
                   THJ2=bend(thumb[0], thumb[1]), THJ1=bend(thumb[1], thumb[2]))
    return targets


def actuator_indices(model, side: str, include_wrist=True) -> dict[str, int]:
    prefix = f"{side}_{side[0]}h_A_"
    result = {model.actuator(i).name[len(prefix):]: i for i in range(model.nu)
              if model.actuator(i).name.startswith(prefix)
              and (include_wrist or model.actuator(i).name[len(prefix):] not in ("WRJ1", "WRJ2"))}
    expected = 20 if include_wrist else 18
    if len(result) != expected:
        raise ValueError(f"Expected {expected} {side} Shadow Hand actuators, got {len(result)}")
    return result


def apply_targets(ctrl, values, indices, ranges):
    for name, value in values.items():
        idx = indices[name]
        ctrl[idx] = np.clip(value, ranges[idx, 0], ranges[idx, 1])


def synthetic_skeleton(angle=0.0):
    """Synthetic geometry for offline tests only; never used as live input."""
    joints = np.repeat(np.eye(4)[None], 25, axis=0)
    for start in (0, 5, 10, 15, 20):
        point = np.array([start * 0.005, 0.0, 0.0])
        for j in range(5):
            joints[start + j, :3, 3] = point
            theta = j * angle
            point = point + 0.025 * np.array([0, np.cos(theta), -np.sin(theta)])
    return joints
