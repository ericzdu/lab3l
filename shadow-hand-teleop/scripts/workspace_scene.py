"""Procedural tabletop props. All distances are metres; mass is kilograms."""
import mujoco
import numpy as np

# Bump SCENE_VERSION whenever the props move; episodes store it in
# metadata.scene, and a missing value means version 1 (SCENE_V1 below).
SCENE_VERSION = 1
SCENE_V1 = {"table_top": .505, "cup_start": (.32, -.28, .562), "tray_center": (.43, -.12)}
TABLE_TOP = .505
# Tray outer half-extents (x, y) and rim wall half-thickness.
TRAY_HALF_EXTENTS = (.10, .075)
TRAY_RIM_THICKNESS = .005
# Cup outer radius and half-height; its free-joint origin is the cup centre.
CUP_RADIUS = .034
CUP_HALF_HEIGHT = .057
CUP_START = np.array([.32, -.28, TABLE_TOP + CUP_HALF_HEIGHT])
TRAY_CENTER = np.array([.43, -.12])


def add_workspace(scene):
    """Keep existing worktop; add legs, fixed tray, and a hollow free cup."""
    for x in (.23, 1.01):
        for y in (-.48, .48):
            scene.worldbody.add_geom(name=f"table_leg_{x}_{y}",
                type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x,y,.19],
                size=[.025,.025,.29], rgba=[.22,.24,.27,1])
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
        rgba=[.9,.65,.22,1],friction=[1,.01,.001])
    segments=24
    for i in range(segments):
        angle=2*np.pi*i/segments
        cup.add_geom(name=f"cup_wall_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[.032*np.cos(angle),.032*np.sin(angle),0],
            quat=[np.cos(angle/2),0,0,np.sin(angle/2)],
            size=[.003,.032*np.tan(np.pi/segments)*1.05,.051],
            mass=.12/segments,rgba=[.95,.72,.28,1],friction=[1,.01,.001])
    # U-shaped handle, formed from three collision capsules.
    for i,(start,end) in enumerate((([.032,0,.032],[.061,0,.032]),
                                  ([.061,0,.032],[.061,0,-.027]),
                                  ([.061,0,-.027],[.032,0,-.027]))):
        cup.add_geom(name=f"cup_handle_{i}",type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*start,*end],size=[.005,0,0],mass=.01,
            rgba=[.95,.72,.28,1],friction=[1,.01,.001])
