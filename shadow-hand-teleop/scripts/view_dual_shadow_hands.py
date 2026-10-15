"""Open one MuJoCo window containing left and right Shadow Hands."""

from __future__ import annotations

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


BASE_SCENE = """
<mujoco model="dual_shadow_hands">
  <compiler angle="radian" autolimits="true"/>
  <option cone="elliptic" impratio="10" timestep="0.002"/>
  <statistic extent="0.7" center="0 0 0.2"/>

  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
    <global azimuth="180" elevation="-25"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1"/>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 -0.1" size="0 0 0.05" type="plane"
      material="groundplane"/>
  </worldbody>
</mujoco>
"""


def build_model(moving=False) -> mujoco.MjModel:
    """Attach the official left and right hand models to one scene."""
    package_path = menagerie.get("shadow_hand").path()
    scene = mujoco.MjSpec.from_string(BASE_SCENE)
    left = mujoco.MjSpec.from_file(str(package_path / "left_hand.xml"))
    right = mujoco.MjSpec.from_file(str(package_path / "right_hand.xml"))

    # The hand models extend along world X, so separate them along world Y.
    # This places the two wrists on the same horizontal line in the default view.
    if moving:
        left_mount = scene.worldbody.add_body(name="left_mount", mocap=True).add_frame()
        right_mount = scene.worldbody.add_body(name="right_mount", mocap=True).add_frame()
    else:
        left_mount = scene.worldbody.add_frame(name="left_mount", pos=[0.0, 0.13, 0.0])
        right_mount = scene.worldbody.add_frame(name="right_mount", pos=[0.0, -0.13, 0.0])

    # Prefixes keep shared mesh, material, and default names from colliding.
    scene.attach(left, prefix="left_", frame=left_mount)
    scene.attach(right, prefix="right_", frame=right_mount)
    if moving:
        probe = scene.worldbody.add_body(name="smoke_probe", mocap=True, pos=[0, 0, -20])
        probe.add_geom(name="smoke_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[.001, 0, 0], rgba=[0, 0, 0, 0], contype=0, conaffinity=0, group=5)
    return scene.compile()


def hand_pose_targets(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return one 20-actuator open pose and fist pose."""
    open_pose = np.zeros(20, dtype=np.float64)
    fist_pose = np.zeros(20, dtype=np.float64)

    fist_pose[2:7] = [0.0, 1.0, 0.0, 0.5, 1.2]
    fist_pose[7:10] = [0.0, 1.3, 2.5]
    fist_pose[10:13] = [0.0, 1.3, 2.5]
    fist_pose[13:16] = [0.0, 1.3, 2.5]
    fist_pose[16:20] = [0.5, 0.0, 1.3, 2.5]

    left_limits = model.actuator_ctrlrange[:20]
    return np.clip(open_pose, left_limits[:, 0], left_limits[:, 1]), np.clip(
        fist_pose, left_limits[:, 0], left_limits[:, 1]
    )


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    hand_open, hand_fist = hand_pose_targets(model)
    state = {"left_fist": False, "right_fist": False}

    def describe_state() -> None:
        left = "fist" if state["left_fist"] else "open"
        right = "fist" if state["right_fist"] else "open"
        print(f"Left: {left} | Right: {right}")

    def on_key(keycode: int) -> None:
        if keycode == glfw.KEY_1:
            state["left_fist"] = not state["left_fist"]
        elif keycode == glfw.KEY_2:
            state["right_fist"] = not state["right_fist"]
        elif keycode == glfw.KEY_F:
            state["left_fist"] = True
            state["right_fist"] = True
        elif keycode == glfw.KEY_O:
            state["left_fist"] = False
            state["right_fist"] = False
        else:
            return
        describe_state()

    print("Opening the dual Shadow Hand simulation.")
    print("Click the viewer before using the keyboard.")
    print("1: toggle left | 2: toggle right | F: both fists | O: both open")
    print("Close the viewer window when you are finished.")

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=on_key
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.18, 0.0, 0.03]
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 180.0
                viewer.cam.elevation = -12.0
            while viewer.is_running():
                target = np.concatenate(
                    [
                        hand_fist if state["left_fist"] else hand_open,
                        hand_fist if state["right_fist"] else hand_open,
                    ]
                )
                with viewer.lock():
                    data.ctrl[:] = target
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
