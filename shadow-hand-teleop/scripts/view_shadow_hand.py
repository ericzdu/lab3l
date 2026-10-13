"""Open the MuJoCo viewer with the simulated Shadow Hand scene."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import glfw
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
    parser = argparse.ArgumentParser(description="Interact with a simulated Shadow Hand.")
    parser.add_argument(
        "--hand",
        choices=("left", "right"),
        default="left",
        help="Which Shadow Hand model to display.",
    )
    return parser.parse_args()


def control_targets(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return neutral and fist actuator targets for either hand model."""
    open_pose = np.zeros(model.nu, dtype=np.float64)
    fist_pose = np.zeros(model.nu, dtype=np.float64)

    # Keep the wrist neutral. Oppose the thumb and flex the four fingers.
    fist_pose[2:7] = [0.0, 1.0, 0.0, 0.5, 1.2]
    fist_pose[7:10] = [0.0, 1.3, 2.5]
    fist_pose[10:13] = [0.0, 1.3, 2.5]
    fist_pose[13:16] = [0.0, 1.3, 2.5]
    fist_pose[16:20] = [0.5, 0.0, 1.3, 2.5]

    limits = model.actuator_ctrlrange
    return np.clip(open_pose, limits[:, 0], limits[:, 1]), np.clip(
        fist_pose, limits[:, 0], limits[:, 1]
    )


def main() -> None:
    args = parse_args()
    model = menagerie.load("shadow_hand", f"scene_{args.hand}")
    data = mujoco.MjData(model)
    open_pose, fist_pose = control_targets(model)
    state = {"make_fist": False}

    def on_key(keycode: int) -> None:
        if keycode == glfw.KEY_F:
            state["make_fist"] = True
            print("F: closing the hand")
        elif keycode == glfw.KEY_O:
            state["make_fist"] = False
            print("O: opening the hand")

    mujoco.mj_forward(model, data)

    print(f"Opening the {args.hand} Shadow Hand simulation.")
    print("Click the viewer, then press F to make a fist or O to open the hand.")
    print("Close the viewer window when you are finished.")

    # On macOS the managed viewer can fail while creating its Cocoa window.
    # The passive viewer is the supported mode for a Python-controlled loop and
    # works with the mjpython launcher used by run_shadow_hand.command.
    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=on_key
        ) as viewer:
            while viewer.is_running():
                with viewer.lock():
                    data.ctrl[:] = fist_pose if state["make_fist"] else open_pose
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
