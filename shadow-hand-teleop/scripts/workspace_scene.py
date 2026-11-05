"""Procedural tabletop props. All distances are metres; mass is kilograms."""
from pathlib import Path

import mujoco
import numpy as np

# Bump SCENE_VERSION whenever the props move; episodes store it in
# metadata.scene, and a missing value means version 1 (SCENE_V1 below).
SCENE_VERSION = 2
SCENE_V1 = {"table_top": .505, "cup_start": (.32, -.28, .562), "tray_center": (.43, -.12)}
# Version 2 raises the worktop 9.5 cm and pulls the cup and tray toward the
# right shoulder so a palm-inward right-hand grasp and placement stay >= 3 cm
# inside HumanArmController's clip box and reach cap (see check_reach.py).
TABLE_TOP = .60
# Tray outer half-extents (x, y) and rim wall half-thickness.
TRAY_HALF_EXTENTS = (.10, .075)
TRAY_RIM_THICKNESS = .005
# Cup outer radius and half-height; its free-joint origin is the cup centre.
CUP_RADIUS = .034
CUP_HALF_HEIGHT = .057
CUP_START = np.array([.30, -.27, TABLE_TOP + CUP_HALF_HEIGHT])
TRAY_CENTER = np.array([.30, -.09])
# Cup contacts: MuJoCo uses the larger condim and the larger friction of the
# two geoms, so these apply to every cup contact (fingers, palm, table, tray).
# condim 4 adds torsional friction, so a cup pinched between thumb and fingers
# does not spin about the contact normal; sliding friction stays 1.
CUP_CONDIM = 4
CUP_FRICTION = [1, .005, .001]
# Worktop is 5 cm thick (human_arms.py); legs run from the floor (z = -0.1,
# BASE_SCENE) to the worktop centre, as before.
FLOOR_Z = -.1
LEG_HALF_HEIGHT = (TABLE_TOP - .025 - FLOOR_Z) / 2


def add_workspace(scene):
    """Keep existing worktop; add legs, fixed tray, and a hollow free cup."""
    for x in (.23, 1.01):
        for y in (-.48, .48):
            scene.worldbody.add_geom(name=f"table_leg_{x}_{y}",
                type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x,y,FLOOR_Z+LEG_HALF_HEIGHT],
                size=[.025,.025,LEG_HALF_HEIGHT], rgba=[.22,.24,.27,1])
    tray=scene.worldbody.add_body(name="tray",pos=[*TRAY_CENTER,TABLE_TOP])
    tray.add_geom(name="tray_base",type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0,0,.006],size=[*TRAY_HALF_EXTENTS,.006],rgba=[.22,.55,.58,1],
        friction=[1,.01,.001])
    for axis in (0,1):
        for sign in (-1,1):
            pos=[0,0,.022]; size=[*TRAY_HALF_EXTENTS,.016]
            pos[axis]=sign*TRAY_HALF_EXTENTS[axis]
            size[axis]=TRAY_RIM_THICKNESS
            tray.add_geom(name=f"tray_rim_{axis}_{sign}",type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,size=size,rgba=[.17,.43,.47,1])
    cup=scene.worldbody.add_body(name="cup",pos=CUP_START)
    cup.add_freejoint(name="cup_free")
    # A real opening: separate convex wall segments, not one solid cylinder.
    cup.add_geom(name="cup_bottom",type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0,0,-(CUP_HALF_HEIGHT-.006)],size=[CUP_RADIUS,.006,0],mass=.06,
        rgba=[.9,.65,.22,1],friction=CUP_FRICTION,condim=CUP_CONDIM)
    segments=24
    for i in range(segments):
        angle=2*np.pi*i/segments
        cup.add_geom(name=f"cup_wall_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[.032*np.cos(angle),.032*np.sin(angle),0],
            quat=[np.cos(angle/2),0,0,np.sin(angle/2)],
            size=[.003,.032*np.tan(np.pi/segments)*1.05,.051],
            mass=.12/segments,rgba=[.95,.72,.28,1],friction=CUP_FRICTION,condim=CUP_CONDIM)
    # U-shaped handle, formed from three collision capsules.
    for i,(start,end) in enumerate((([.032,0,.032],[.061,0,.032]),
                                  ([.061,0,.032],[.061,0,-.027]),
                                  ([.061,0,-.027],[.032,0,-.027]))):
        cup.add_geom(name=f"cup_handle_{i}",type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*start,*end],size=[.005,0,0],mass=.01,
            rgba=[.95,.72,.28,1],friction=CUP_FRICTION,condim=CUP_CONDIM)
    # Grasp-assist welds (grasp_assist.py), one per hand, inactive unless
    # --grasp-assist switches one on. Present in every workspace model so
    # nq/nv/nu are the same with or without the assist.
    for side in ("left", "right"):
        palm = f"{side}_{side[0]}h_palm"
        if scene.body(palm) is not None:
            scene.add_equality(name=f"{side}_grasp_assist", type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY, name1=palm, name2="cup", active=False,
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])


# Optional free props (view_dual_arms.py --props). They are NOT part of the
# default scene: each adds a free joint, and replay requires the recorded joint
# list to match, so cup_to_tray episodes keep replaying unchanged. Episodes
# store the chosen names in metadata.scene.props and replay rebuilds with them,
# so SCENE_VERSION does not change when a prop is added here.
# Keys: type (box/sphere/cylinder/capsule/mesh), size (MuJoCo half-sizes),
# xy (spawn on the worktop, it rests at TABLE_TOP + half height), mass, rgba.
# The left half of the table is free and inside the left hand's reach
# (HumanArmController clips wrist targets to home +/- 16 cm in x); the worktop's
# front edge is x = 0.16, so keep a prop's footprint behind it.
# Mesh props: put an .obj/.stl in shadow-hand-teleop/assets/ and use
# type "mesh", mesh "<file>", half_height (for the spawn z), optional scale.
# MuJoCo collides with the mesh's convex hull, so hollow shapes fill in.
PROPS_DIR = Path(__file__).resolve().parents[1] / "assets"
PROPS = {
    "block":  {"type": "box", "size": (.025, .025, .025), "xy": (.30, .27), "mass": .05,
               "rgba": (.80, .25, .22, 1)},
    "bottle": {"type": "cylinder", "size": (.03, .08), "xy": (.32, .10), "mass": .15,
               "rgba": (.30, .55, .85, 1)},
    "plate":  {"type": "cylinder", "size": (.07, .006), "xy": (.23, .18), "mass": .10,
               "rgba": (.92, .92, .88, 1)},
    "apple":  {"type": "sphere", "size": (.035,), "xy": (.24, .04), "mass": .15,
               "rgba": (.75, .15, .15, 1)},
}
_PROP_GEOMS = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
               "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
               "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "mesh": mujoco.mjtGeom.mjGEOM_MESH}
# Same contact model as the cup. Rolling friction (condim 6) keeps the apple
# from rolling off the table on its own.
PROP_FRICTION = [1, .005, .001]


def prop_half_height(spec):
    if spec["type"] == "mesh":
        return spec["half_height"]
    size = spec["size"]
    return {"box": lambda: size[2], "sphere": lambda: size[0], "cylinder": lambda: size[1],
            "capsule": lambda: size[0] + size[1]}[spec["type"]]()


def prop_start(name):
    spec = PROPS[name]
    return np.array([*spec["xy"], TABLE_TOP + prop_half_height(spec)])


def add_props(scene, names):
    """Add the named PROPS as free bodies, in PROPS order (stable joint order)."""
    unknown = sorted(set(names) - PROPS.keys())
    if unknown:
        raise ValueError(f"Unknown prop(s) {unknown}; choose from {sorted(PROPS)}")
    for name in (n for n in PROPS if n in names):
        spec = PROPS[name]
        body = scene.worldbody.add_body(name=name, pos=prop_start(name))
        body.add_freejoint(name=f"{name}_free")
        geom = body.add_geom(name=f"{name}_geom", type=_PROP_GEOMS[spec["type"]],
            rgba=list(spec["rgba"]), mass=spec["mass"], friction=PROP_FRICTION,
            condim=6 if spec["type"] == "sphere" else CUP_CONDIM)
        if spec["type"] == "mesh":
            mesh = scene.add_mesh(name=f"{name}_mesh", file=str(PROPS_DIR / spec["mesh"]),
                                  scale=list(spec.get("scale", (1, 1, 1))))
            geom.meshname = mesh.name
        else:
            geom.size = [*spec["size"], *[0] * (3 - len(spec["size"]))]
