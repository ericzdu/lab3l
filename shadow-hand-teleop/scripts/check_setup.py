"""Verify that MuJoCo and the Shadow Hand model run correctly."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import mujoco_menagerie as menagerie
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MENAGERIE_CACHE_DIR",
    str(PROJECT_ROOT / ".cache" / "mujoco_menagerie"),
)


def main() -> None:
    model = menagerie.load("shadow_hand")
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    for _ in range(250):
        mujoco.mj_step(model, data)

    if not np.isfinite(data.qpos).all():
        raise RuntimeError("Simulation produced invalid joint positions.")

    print("Setup check passed.")
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Shadow Hand: {model.nu} actuators, {model.nq} position coordinates")
    print(f"Scene: {model.nbody} bodies, {model.ngeom} collision/visual geometries")


if __name__ == "__main__":
    main()

