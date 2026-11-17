"""Real 3D object assets: load an object folder into a MuJoCo spec.

An object folder (made by prepare_object.py) holds:

    object.json            physics, size, footprint, placement, source/license
    visual_<i>.obj (.png)  look only: no contacts, no mass (geom group 2)
    collision/hull_*.obj   convex pieces carrying all contacts and mass (group 3)
    source/<file>          optional original export (e.g. a SAM 3D GLB)

Meshes are metres, Z-up, with the object's bottom centre at the origin. When
added to a body, geoms are shifted down by half the object's height, so the
BODY origin is the centre of the bounding box: workspace props (and the
episode labeler) treat a body's position as its centre, resting at
table_top + half height.

Loading uses MuJoCo only (MuJoCo reads OBJ vertices, faces and UVs itself);
trimesh/CoACD are needed only to prepare a folder, not to run the sim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

OBJECTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "objects"
SCHEMA = "lab3l.object"
SCHEMA_VERSION = 1
VISUAL_GROUP, COLLISION_GROUP = 2, 3
MAX_HULLS = 64
MAX_BOXES = 128
SOURCE_KINDS = ("sam3d_export", "scanned_dataset", "external_mesh", "synthetic_test")
VALUE_ORIGINS = ("measured", "assumed", "scaled_placeholder", "dataset")
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,47}")


def _vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def _positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def _inside(root, name):
    """Asset paths are relative and cannot escape the object folder."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Asset paths must be nonempty relative paths inside the object folder")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or path outside object folder: {name}")
    return path


def resolve(asset):
    """Object folder for an asset id (under assets/objects) or a path."""
    path = Path(asset).expanduser()
    if path.name == "object.json":
        path = path.parent
    if not path.is_absolute() and not path.exists() and ID_PATTERN.fullmatch(str(asset)):
        path = OBJECTS_DIR / str(asset)
    path = path.resolve()
    if not (path / "object.json").is_file():
        raise ValueError(f"No object.json in {path}")
    return path


_CACHE = {}


def load(asset):
    """Validated object.json (with absolute file paths added), cached per folder."""
    root = resolve(asset)
    stamp = (root / "object.json").stat().st_mtime_ns
    if root in _CACHE and _CACHE[root][0] == stamp:
        return _CACHE[root][1]
    config = json.loads((root / "object.json").read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("version") != SCHEMA_VERSION:
        raise ValueError(f"{root}: not a {SCHEMA} v{SCHEMA_VERSION} object.json")
    if not ID_PATTERN.fullmatch(str(config.get("id", ""))):
        raise ValueError(f"{root}: id must start with a letter; letters, digits, '_' (max 48)")
    size = _vector(config["size_m"], 3, "size_m")
    if np.any(size <= 0) or np.max(size) > 3:
        raise ValueError(f"{root}: size_m must be positive and at most 3 m (check units)")
    _positive(config["mass_kg"], "mass_kg")
    contact = config.get("contact", {})
    friction = _vector(contact.get("friction", [1, .005, .001]), 3, "contact.friction")
    if np.any(friction < 0):
        raise ValueError("Friction cannot be negative")
    if int(contact.get("condim", 4)) not in (1, 3, 4, 6):
        raise ValueError("contact.condim must be 1, 3, 4 or 6")
    visuals = config.get("visual", [])
    if not isinstance(visuals, list) or not 1 <= len(visuals) <= 64:
        raise ValueError("visual needs 1-64 mesh parts")
    for part in visuals:
        part["_mesh"] = _inside(root, part["mesh"])
        part["_texture"] = _inside(root, part["texture"]) if part.get("texture") else None
        rgba = _vector(part.get("rgba", [.7, .7, .7, 1]), 4, "rgba")
        if np.any(rgba < 0) or np.any(rgba > 1):
            raise ValueError("rgba must be in [0, 1]")
    collision = config.get("collision", {})
    mode = collision.get("mode")
    if mode == "hulls":
        hulls = collision.get("hulls", [])
        if not 1 <= len(hulls) <= MAX_HULLS:
            raise ValueError(f"hulls mode needs 1-{MAX_HULLS} convex pieces")
        for hull in hulls:
            hull["_mesh"] = _inside(root, hull["mesh"])
            _positive(hull["volume_m3"], "hull volume_m3")
    elif mode == "boxes":
        boxes = collision.get("boxes", [])
        if not 1 <= len(boxes) <= MAX_BOXES:
            raise ValueError(f"boxes mode needs 1-{MAX_BOXES} boxes")
        for box in boxes:
            _vector(box["position_m"], 3, "collision box position_m")
            if np.any(_vector(box["half_size_m"], 3, "collision box half_size_m") <= 0):
                raise ValueError("collision box sizes must be positive")
    else:
        raise ValueError("collision.mode must be hulls or boxes")
    footprint = config.get("footprint", {})
    if footprint.get("shape") == "round":
        _positive(footprint["radius_m"], "footprint radius_m")
    elif footprint.get("shape") == "box":
        if np.any(_vector(footprint["half_extents_m"], 2, "footprint half_extents_m") <= 0):
            raise ValueError("footprint half extents must be positive")
    else:
        raise ValueError("footprint.shape must be round or box")
    if config.get("source", {}).get("kind") not in SOURCE_KINDS:
        raise ValueError(f"source.kind must be one of {SOURCE_KINDS}")
    config["_root"] = root
    _CACHE[root] = (stamp, config)
    return config


def half_height(asset):
    return float(load(asset)["size_m"][2]) / 2


def footprint(asset):
    """("round", radius) or ("box", (half_x, half_y)) in the object's frame."""
    shape = load(asset)["footprint"]
    if shape["shape"] == "round":
        return "round", float(shape["radius_m"])
    return "box", tuple(float(v) for v in shape["half_extents_m"])


def digest(asset):
    """sha256 over every file in the object folder (relative path + bytes)."""
    root = resolve(asset)
    sha = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        sha.update(path.relative_to(root).as_posix().encode() + b"\0")
        sha.update(path.read_bytes())
    return sha.hexdigest()


def summary(asset):
    """What an episode stores about the model it used."""
    config = load(asset)
    source = config["source"]
    return {"id": config["id"], "sha256": digest(asset), "source_kind": source["kind"],
            "source_name": source.get("name"), "license": source.get("license")}


def add_geoms(scene, body, asset, prefix, rgba_override=None):
    """Visual + collision geoms (and their meshes/textures) on `body`.

    Collision pieces share mass_kg by volume; MuJoCo then derives the body's
    inertia from them. Returns the collision geoms.
    """
    config = load(asset)
    shift = [0, 0, -config["size_m"][2] / 2]
    for index, part in enumerate(config["visual"]):
        name = f"{prefix}_visual_{index}"
        # Scans are rarely watertight; the convex inertia mode never fails on
        # them, and a visual geom has no mass anyway.
        scene.add_mesh(name=name, file=str(part["_mesh"]),
                       inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
        kwargs = {}
        rgba = list(rgba_override or part.get("rgba", [.7, .7, .7, 1]))
        if part["_texture"] is not None:
            scene.add_texture(name=name + "_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
                              file=str(part["_texture"]))
            slots = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
            slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = name + "_texture"
            scene.add_material(name=name + "_material", textures=slots, rgba=[1, 1, 1, 1])
            kwargs["material"] = name + "_material"
            rgba = [1, 1, 1, 1]
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=shift,
                      contype=0, conaffinity=0, mass=0, density=0, group=VISUAL_GROUP,
                      rgba=rgba, **kwargs)
    contact = config.get("contact", {})
    common = dict(friction=list(contact.get("friction", [1, .005, .001])),
                  condim=int(contact.get("condim", 4)), group=COLLISION_GROUP,
                  rgba=[.1, .9, .4, .3])
    collision, mass = config["collision"], float(config["mass_kg"])
    geoms = []
    if collision["mode"] == "hulls":
        volumes = np.array([float(h["volume_m3"]) for h in collision["hulls"]])
        for index, (hull, share) in enumerate(zip(collision["hulls"], volumes / volumes.sum())):
            name = f"{prefix}_hull_{index}"
            scene.add_mesh(name=name, file=str(hull["_mesh"]))
            geoms.append(body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                       pos=shift, mass=mass * share, **common))
    else:
        volumes = np.array([np.prod(box["half_size_m"]) for box in collision["boxes"]])
        for index, (box, share) in enumerate(zip(collision["boxes"], volumes / volumes.sum())):
            geoms.append(body.add_geom(name=f"{prefix}_box_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                       pos=np.add(box["position_m"], shift).tolist(),
                                       size=list(box["half_size_m"]), mass=mass * share, **common))
    return geoms


def add_body(scene, asset, name, position_center, yaw_degrees=0., fixed=False):
    """A body named `name` whose origin is the object's bounding-box centre."""
    angle = np.deg2rad(yaw_degrees) / 2
    body = scene.worldbody.add_body(name=name, pos=list(position_center),
                                    quat=[np.cos(angle), 0, 0, np.sin(angle)])
    if not fixed:
        body.add_freejoint(name=f"{name}_free")
    add_geoms(scene, body, asset, name)
    return body
