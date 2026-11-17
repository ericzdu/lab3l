"""Realistic-looking visual meshes for the workspace props (look only, no physics).

Each prop keeps its simple collision shape from workspace_scene.PROPS (sphere,
cylinder, box), which the reach checks and scripted demos were tuned on. This
module adds a separate visual geom on top: contype/conaffinity 0 and mass 0,
so it never touches anything and does not change the prop's weight or inertia.
The collision geom moves to group 3, which the viewer and recorded videos
hide by default (the same split the Shadow Hand model uses).

Shapes are surfaces of revolution built in code from a 2D outline, so no mesh
files or downloads are needed. Outlines are (radius, z) points in metres in the
prop's body frame, traced bottom to top around the outside; the first and last
points sit on the axis (radius 0) to close the surface.
"""
from __future__ import annotations

import mujoco
import numpy as np

SEGMENTS = 40                 # facets around the axis
VISUAL_GROUP, COLLISION_GROUP = 2, 3


def lathe(profile, segments=SEGMENTS):
    """(vertices, faces) of the closed surface swept by `profile` around +z.

    A profile that never touches the axis must end where it started (a tube).

    Faces wind counter-clockwise seen from outside: normal = tangent x
    profile direction, which points out of the solid for any outline traced
    bottom to top around the outside, convex or not (e.g. a plate's well).
    """
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    vertices, rings = [], []
    closed = len(profile) > 2 and tuple(profile[0]) == tuple(profile[-1]) and profile[0][0] > 1e-9
    for r, z in (profile[:-1] if closed else profile):
        if r <= 1e-9:                      # pole: one vertex on the axis
            rings.append([len(vertices)] * segments)
            vertices.append((0., 0., z))
        else:
            rings.append(list(range(len(vertices), len(vertices) + segments)))
            vertices.extend((r * np.cos(a), r * np.sin(a), z) for a in angles)
    if closed:
        rings.append(rings[0])
    faces = []
    for lower, upper in zip(rings, rings[1:]):
        for j in range(segments):
            k = (j + 1) % segments
            a, b, c, d = lower[j], lower[k], upper[k], upper[j]
            if a != b:                     # lower ring is not a pole
                faces.append((a, b, c))
            if c != d:                     # upper ring is not a pole
                faces.append((a, c, d))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def apple_profile(radius, points=24):
    """Slightly squat sphere, wider near the top, with dimples at both poles."""
    profile = []
    for t in np.linspace(0, np.pi, points):
        r = radius * np.sin(t) * (1 + .07 * np.cos(np.pi - t))
        z = -radius * np.cos(t) * .94
        z += .22 * radius * np.exp(-(t / .35) ** 2)            # bottom dimple
        z -= .30 * radius * np.exp(-((np.pi - t) / .40) ** 2)  # stem well
        profile.append((max(r, 0.), z))
    profile[0], profile[-1] = (0., profile[0][1]), (0., profile[-1][1])
    return profile


def bottle_profile(radius, half_height):
    """Glass bottle filling the collision cylinder: body, shoulder, neck, lip."""
    h, r = half_height, radius
    neck = .42 * r
    return [(0, -h), (.93 * r, -h), (r, -h + .004), (r, .25 * h), (.92 * r, .45 * h),
            (.62 * r, .60 * h), (neck, .70 * h), (neck, .83 * h), (neck * 1.15, .85 * h),
            (neck * 1.15, .87 * h), (0, .87 * h)]


def plate_profile(radius, half_thickness):
    """Ceramic plate: foot ring, shallow well, raised rim, inside the collision disc."""
    t, r = half_thickness, radius
    return [(0, -t + .002), (.62 * r, -t + .002), (.66 * r, -t), (.72 * r, -t), (.75 * r, -t + .002),
            (r, t * .55), (r, t * 1.45), (.95 * r, t * 1.55), (.85 * r, t * 1.1),
            (.66 * r, t * .85), (0, t * .85)]


def _mesh(scene, name, profile):
    vertices, faces = lathe(profile)
    scene.add_mesh(name=name, uservert=vertices.ravel().tolist(), userface=faces.ravel().tolist())
    return name


def _material(scene, name, rgba, specular=.3, shininess=.3, reflectance=0.):
    scene.add_material(name=name, rgba=list(rgba), specular=specular, shininess=shininess,
                       reflectance=reflectance)
    return name


def _visual(body, name, **kwargs):
    return body.add_geom(name=name, contype=0, conaffinity=0, group=VISUAL_GROUP, mass=0, **kwargs)


def add_apple(scene, body, spec):
    radius = spec["size"][0]
    mesh = _mesh(scene, "apple_visual", apple_profile(radius))
    _visual(body, "apple_skin", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
            material=_material(scene, "apple_skin", (.72, .09, .08, 1), specular=.6, shininess=.6))
    # Short woody stem, tilted, sitting in the stem well.
    top = radius * .94 - .30 * radius
    _visual(body, "apple_stem", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[.0022, 0, 0],
            fromto=[0, 0, top, .004, .002, top + .016],
            material=_material(scene, "apple_stem", (.36, .23, .12, 1), specular=.05))


def add_bottle(scene, body, spec):
    radius, half = spec["size"]
    mesh = _mesh(scene, "bottle_visual", bottle_profile(radius, half))
    _visual(body, "bottle_glass", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
            material=_material(scene, "bottle_glass", (.22, .55, .38, .55), specular=.9,
                               shininess=.9, reflectance=.1))
    # Paper label around the body and a screw cap on the neck.
    # The label is a thin tube just outside the glass, so no end face shows through it.
    inner, outer, top, bottom = radius * 1.008, radius * 1.014, .02 * half, -.42 * half
    label = _mesh(scene, "bottle_label", [(inner, bottom), (outer, bottom), (outer, top),
                                          (inner, top), (inner, bottom)])
    _visual(body, "bottle_label", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=label,
            material=_material(scene, "bottle_label", (.93, .9, .8, 1), specular=.1))
    _visual(body, "bottle_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[.42 * radius * 1.25, .065 * half, 0], pos=[0, 0, .935 * half],
            material=_material(scene, "bottle_cap", (.15, .3, .7, 1), specular=.4))


def add_plate(scene, body, spec):
    radius, half = spec["size"]
    mesh = _mesh(scene, "plate_visual", plate_profile(radius, half))
    _visual(body, "plate_ceramic", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
            material=_material(scene, "plate_ceramic", (.95, .95, .92, 1), specular=.7,
                               shininess=.8, reflectance=.05))


def add_block(scene, body, spec, name="block", inset_rgba=(.62, .16, .12, 1)):
    # A wooden toy block: same box, wood colour, with a darker inset on each
    # side face so its orientation reads in video.
    size = np.asarray(spec["size"], dtype=float)
    _visual(body, f"{name}_wood", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size),
            material=_material(scene, f"{name}_wood", (.78, .58, .34, 1), specular=.15))
    inset = _material(scene, f"{name}_inset", inset_rgba, specular=.15)
    for axis in (0, 1):
        for sign in (-1, 1):
            pos = [0., 0., 0.]
            pos[axis] = sign * size[axis]
            half = list(size * .62)
            half[axis] = .0006
            _visual(body, f"{name}_inset_{axis}_{'np'[sign > 0]}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=half, pos=pos, material=inset)


def add_big_block(scene, body, spec):
    add_block(scene, body, spec, "big_block", inset_rgba=(.16, .32, .62, 1))


VISUALS = {"apple": add_apple, "bottle": add_bottle, "plate": add_plate, "block": add_block,
           "big_block": add_big_block}


def add_visual(scene, body, name, spec, collision):
    """Dress prop `name`; hide its collision geom if a visual was added."""
    builder = VISUALS.get(name)
    if builder is None:
        return False
    builder(scene, body, spec)
    collision.group = COLLISION_GROUP
    return True
