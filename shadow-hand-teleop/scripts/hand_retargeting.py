"""Conservative, uncalibrated finger-flexion and thumb-opposition mapping.

Uses VisionProTeleop's 25-joint wrist-local skeleton. This is not full-pose IK:
finger spread, thumb rotation (THJ5), THJ3 and wrist/arm motion are not
reconstructed. Thumb opposition (THJ4) is estimated from how far the thumb
metacarpal leaves the palm plane, plus a small grip coupling toward the
keyboard fist's THJ4 as the fingers close.
"""

from __future__ import annotations

import numpy as np


def bend(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-8:
        raise ValueError("Missing or degenerate finger tracking")
    return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1)))


# Thumb metacarpal angle out of the palm plane (degrees) mapped linearly onto
# THJ4's range [0, THJ4_MAX]. From recorded live episodes: about 5-20 degrees
# with a relaxed open hand, 35-48 degrees with the thumb across a grasp.
OPPOSITION_DEGREES = (10., 45.)
THJ4_MAX = 1.22          # Shadow THJ4 upper limit (rad)
# Grip coupling: as mean finger closure (0 open .. 1 fully flexed) rises from
# GRIP_RANGE[0] to [1], THJ4 is raised toward the keyboard fist's value.
GRIP_RANGE = (.35, .7)
FIST_THJ4 = 1.0


def thumb_opposition(positions: np.ndarray) -> float:
    """0 (thumb in the palm plane) .. 1 (fully opposed), from joint positions.

    Uses the unsigned angle, so mirrored (left/right) skeletons agree; a
    thumb bent behind the palm is anatomically small and reads as low.
    """
    wrist = positions[0]
    normal = np.cross(positions[6] - wrist, positions[21] - wrist)   # index and little knuckles
    metacarpal = positions[2] - positions[1]
    denominator = np.linalg.norm(normal) * np.linalg.norm(metacarpal)
    if denominator < 1e-12:
        raise ValueError("Missing or degenerate palm tracking")
    angle = np.degrees(np.arcsin(min(1., abs(np.dot(normal, metacarpal)) / denominator)))
    low, high = OPPOSITION_DEGREES
    return float(np.clip((angle - low) / (high - low), 0, 1))


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
    thj4 = THJ4_MAX * thumb_opposition(positions)
    # Mean closure as a fraction of the J3 (pi/2) and J0 (pi) ranges.
    closure = np.mean([(targets[f"{p}J3"] / (np.pi / 2) + targets[f"{p}J0"] / np.pi) / 2
                       for p in ("FF", "MF", "RF", "LF")])
    grip = np.clip((closure - GRIP_RANGE[0]) / (GRIP_RANGE[1] - GRIP_RANGE[0]), 0, 1)
    thj4 += grip * max(0., FIST_THJ4 - thj4)
    targets.update(THJ5=0.0, THJ4=float(min(thj4, THJ4_MAX)), THJ3=0.0,
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


def _rotate(vector, axis, angle):
    """Rodrigues rotation of `vector` about unit `axis`."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle)))


def synthetic_grasp_skeleton(closure=0.0):
    """Synthetic human hand, open (0) to a fist with the thumb across (1).

    Offline tests only; never used as live input. Wrist at the origin, palm in
    the x-y plane, fingers along +y, palmar side -z, thumb on the +x side.
    Rough adult proportions (palm ~9 cm, as in the recorded episodes).
    """
    c = float(np.clip(closure, 0, 1))
    joints = np.repeat(np.eye(4)[None], 25, axis=0)
    # Fingers: metacarpal base, knuckle, then three phalanges (VisionProTeleop order).
    for start, x, length in ((5, .022, .045), (10, .002, .048), (15, -.016, .045), (20, -.032, .036)):
        points = [np.array([x * .4, .01, 0]), np.array([x, .09, 0])]
        direction = np.array([0., 1, 0])
        for flex, bone in zip((1.3, 1.5, 1.0), (length, length * .62, length * .5)):
            direction = _rotate(direction, [1, 0, 0], -flex * c)
            points.append(points[-1] + bone * direction)
        joints[start:start + 5, :3, 3] = points
    # Thumb: CMC near the wrist; the metacarpal swings out of the palm plane
    # (opposition) and across toward the fingers, then flexes at MCP and IP.
    base = np.array([.025, .02, 0])
    across = _rotate(np.array([.6, .8, 0]), [0, 0, 1], np.deg2rad(35 * c))   # toward the fingers
    elevation = np.deg2rad(5 + 45 * c)                                         # out of plane, palmar
    direction = np.cos(elevation) * across + np.sin(elevation) * np.array([0, 0, -1.])
    points = [base]
    hinge = np.cross(direction, [0, 0, -1])
    for flex, bone in zip((0, .5, .8), (.045, .035, .028)):
        direction = _rotate(direction, hinge, flex * c)
        points.append(points[-1] + bone * direction)
    joints[1:5, :3, 3] = points
    return joints


# Fixed key order of finger_targets(); recordings store targets in this order.
FINGER_TARGET_NAMES = tuple(finger_targets(synthetic_skeleton()))
