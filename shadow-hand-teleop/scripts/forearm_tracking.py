"""Decode optional wrist-local SDK forearm points without trusting placeholders."""
import numpy as np
from arm_control import wrist_transform


def forearm_vector(arm, wrist):
    """Elbow-side -> wrist-side vector, in the SDK world frame (metres).

    avp_stream's dict `*_arm` is wrist-local, unlike HandData attributes.
    Missing/invalid optional forearm data must never discard valid fingers.
    A plausible vector is not proof of sensor confidence: this SDK does not
    expose per-joint tracking flags in these arrays.
    """
    if arm is None:
        return None
    try:
        arm = np.asarray(arm, dtype=float)
        if arm.ndim != 3 or arm.shape[0] < 27 or arm.shape[1:] != (4, 4):
            return None
        wrist_end = wrist_transform(arm[25])
        elbow_end = wrist_transform(arm[26])
        local = wrist_end[:3, 3] - elbow_end[:3, 3]
        length = np.linalg.norm(local)
        if not .08 <= length <= .50:
            return None
        return wrist_transform(wrist)[:3, :3] @ local
    except (ValueError, TypeError):
        return None


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
        raise ValueError("Expected a finite nonzero 3D direction")
    return vector / np.linalg.norm(vector)
