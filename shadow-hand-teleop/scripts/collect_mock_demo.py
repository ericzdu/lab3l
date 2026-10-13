"""Generate a mock Shadow Hand demonstration before Vision Pro is available."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import mujoco_menagerie as menagerie
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MENAGERIE_CACHE_DIR",
    str(PROJECT_ROOT / ".cache" / "mujoco_menagerie"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an open-close Shadow Hand motion and save robot-ready data."
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the MuJoCo viewer and run in real time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "mock_grasp_demo.npz",
        help="Destination .npz dataset path.",
    )
    return parser.parse_args()


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def grasp_amount(t: float) -> float:
    """Open for 1 s, close for 2 s, hold 1 s, and reopen for 2 s."""
    if t < 1.0:
        return 0.0
    if t < 3.0:
        return smoothstep((t - 1.0) / 2.0)
    if t < 4.0:
        return 1.0
    return 1.0 - smoothstep((t - 4.0) / 2.0)


def control_targets(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    open_pose = np.zeros(model.nu, dtype=np.float64)
    close_pose = np.zeros(model.nu, dtype=np.float64)

    # Wrist remains neutral. Thumb opposes the four fingers.
    close_pose[2:7] = [0.0, 1.0, 0.0, 0.5, 1.2]

    # Finger abduction stays neutral while the flexion actuators close.
    close_pose[7:10] = [0.0, 1.3, 2.5]
    close_pose[10:13] = [0.0, 1.3, 2.5]
    close_pose[13:16] = [0.0, 1.3, 2.5]
    close_pose[16:20] = [0.5, 0.0, 1.3, 2.5]

    limits = model.actuator_ctrlrange
    return np.clip(open_pose, limits[:, 0], limits[:, 1]), np.clip(
        close_pose, limits[:, 0], limits[:, 1]
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = menagerie.load("shadow_hand")
    data = mujoco.MjData(model)
    open_pose, close_pose = control_targets(model)

    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    duration = 6.0
    sample_period = 1.0 / 50.0
    next_sample = 0.0

    timestamps: list[float] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    object_pose: list[np.ndarray] = []
    contact_count: list[int] = []

    viewer_context = (
        mujoco.viewer.launch_passive(model, data) if args.viewer else nullcontext()
    )

    with viewer_context as viewer:
        while data.time < duration:
            alpha = grasp_amount(data.time)
            data.ctrl[:] = (1.0 - alpha) * open_pose + alpha * close_pose
            mujoco.mj_step(model, data)

            if data.time + 1e-9 >= next_sample:
                timestamps.append(float(data.time))
                qpos.append(data.qpos.copy())
                qvel.append(data.qvel.copy())
                actions.append(data.ctrl.copy())
                object_pose.append(
                    np.concatenate(
                        [data.xpos[object_body].copy(), data.xquat[object_body].copy()]
                    )
                )
                contact_count.append(int(data.ncon))
                next_sample += sample_period

            if args.viewer:
                viewer.sync()
                time.sleep(model.opt.timestep)

    np.savez_compressed(
        args.output,
        timestamp=np.asarray(timestamps),
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        action=np.asarray(actions),
        object_pose=np.asarray(object_pose),
        contact_count=np.asarray(contact_count),
    )

    metadata = {
        "source": "mock_input",
        "simulator": f"MuJoCo {mujoco.__version__}",
        "robot": "Shadow Hand E3M5 (left)",
        "task": "open-close grasp motion",
        "task_outcome": "not_evaluated",
        "sample_rate_hz": 50,
        "duration_seconds": duration,
        "samples": len(timestamps),
        "fields": [
            "timestamp",
            "qpos",
            "qvel",
            "action",
            "object_pose",
            "contact_count",
        ],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Saved {len(timestamps)} samples to {args.output}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Maximum simultaneous contacts: {max(contact_count, default=0)}")


if __name__ == "__main__":
    main()

