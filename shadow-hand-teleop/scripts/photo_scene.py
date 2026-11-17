"""--scene-config: place imported 3D objects in the human-arm workspace.

Accepts
  * an object folder (or its object.json) made by prepare_object.py, placed at
    the object's own default placement;
  * a scene.json, version 2: objects that each reference an object folder,
    with their own placement (see docs/ADDING_OBJECTS.md);
  * a legacy version-1 bundle (the original SAM 3D import prototype), still
    loaded by the code below with its box/convex/boxes collision modes.

Every imported body is named photo_<id>. Episodes record imported objects like
props and store the scene's path and sha256, so replay can rebuild it.
This is NOT a SAM 3D inference implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

import object_assets
from prepare_object import mesh_library, normalize_parts, triangle_parts  # noqa: F401 (legacy API)

MAX_FACES = 100_000
MAX_PARTS = 64
MAX_OBJECTS = 32
PREFIX = "photo_"


def vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value):
        raise ValueError("Object id must start with a letter and use letters, digits or underscores (max 48)")
    return value


def local_asset(root, name):
    """Bundle paths cannot escape their containing directory or fetch URLs."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Asset paths must be nonempty relative paths within the scene bundle")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing asset or path outside scene bundle: {name}")
    return path



def load_bundle(config_path):
    """Legacy version-1 bundle (Emma's original format): validate before building."""
    path = Path(config_path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Unsupported scene bundle version")
    objects = config.get("objects", [])
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"Scene requires 1–{MAX_OBJECTS} objects")
    result, names, total_faces = [], set(), 0
    for obj in objects:
        name = identifier(obj["id"])
        if name in names:
            raise ValueError(f"Duplicate object id: {name}")
        names.add(name)
        position = vector(obj["position_m"], 3, "position_m")
        yaw = float(obj.get("yaw_degrees", 0))
        if not np.isfinite(yaw):
            raise ValueError("yaw_degrees must be finite")
        fixed = obj.get("fixed", False)
        if not isinstance(fixed, bool):
            raise ValueError("fixed must be true or false")
        mass = positive(obj.get("mass_kg", .15), "mass_kg")
        friction = vector(obj.get("friction", [1., .01, .001]), 3, "friction")
        if np.any(friction < 0):
            raise ValueError("Friction cannot be negative")
        records = obj.get("parts", [])
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_PARTS:
            raise ValueError("Object needs 1–64 mesh parts")
        parts = []
        for record in records:
            meshes = triangle_parts(local_asset(path.parent, record["mesh"]))
            if len(meshes) != 1:
                raise ValueError("Prepared visual files must contain exactly one mesh each")
            mesh = meshes[0]
            total_faces += len(mesh.faces)
            rgba = vector(record.get("rgba", [.7, .6, .4, 1.]), 4, "rgba")
            if np.any(rgba < 0) or np.any(rgba > 1):
                raise ValueError("RGBA must be in [0, 1]")
            texture = local_asset(path.parent, record["texture"]) if record.get("texture") else None
            if texture:
                uv = getattr(mesh.visual, "uv", None)
                if uv is None or np.shape(uv) != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
                    raise ValueError("Textured mesh requires finite per-vertex UVs")
            parts.append((mesh, rgba, texture))
        if total_faces > MAX_FACES:
            raise ValueError(f"Scene exceeds total budget of {MAX_FACES} triangles")
        vertices = np.concatenate([mesh.vertices for mesh, _, _ in parts])
        if np.max(np.ptp(vertices, axis=0)) > 5:
            raise ValueError("Prepared object exceeds 5 m; check units/scale")
        collision = obj.get("collision", {"mode": "box"})
        mode = collision.get("mode")
        boxes = []
        hull = None
        if mode == "box":
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            boxes = [((low + high) / 2, (high - low) / 2)]
        elif mode == "convex":
            hull = mesh_library().Trimesh(vertices=vertices, process=False).convex_hull
        elif mode == "boxes":
            for box in collision.get("boxes", []):
                center = vector(box["position_m"], 3, "collision box position")
                size = vector(box["half_size_m"], 3, "collision box half-size")
                if np.any(size <= 0):
                    raise ValueError("Collision box sizes must be positive")
                boxes.append((center, size))
            if not 1 <= len(boxes) <= 128:
                raise ValueError("boxes mode requires 1–128 manually fitted boxes")
        else:
            raise ValueError("Collision mode must be box, convex or boxes")
        result.append(dict(name=name, position=position, yaw=yaw, fixed=fixed, mass=mass,
            friction=friction, parts=parts, boxes=boxes, hull=hull, mode=mode,
            provenance=obj.get("provenance", {}).get("kind", "unspecified")))
    return result


def _add_legacy_scene(spec, config_path):
    """Visual mesh has no contacts/mass; explicit collision proxy carries both."""
    objects = load_bundle(config_path)
    for obj in objects:
        prefix = "photo_" + obj["name"]
        angle = np.deg2rad(obj["yaw"]) / 2
        body = spec.worldbody.add_body(name=prefix, pos=obj["position"],
            quat=[np.cos(angle), 0, 0, np.sin(angle)])
        if not obj["fixed"]:
            body.add_freejoint(name=prefix + "_free")
        for index, (mesh, rgba, texture) in enumerate(obj["parts"]):
            asset = f"{prefix}_visual_{index}"
            kwargs = {}
            if texture:
                uv = np.asarray(mesh.visual.uv)
                kwargs.update(usertexcoord=uv.ravel(), userfacetexcoord=mesh.faces.ravel())
                spec.add_texture(name=asset + "_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
                    file=str(texture))
                slots = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
                slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = asset + "_texture"
                spec.add_material(name=asset + "_material", textures=slots, rgba=rgba)
                kwargs["material"] = asset + "_material"
            spec.add_mesh(name=asset, uservert=mesh.vertices.ravel(), userface=mesh.faces.ravel(),
                inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
                **{k: v for k, v in kwargs.items() if k != "material"})
            body.add_geom(name=asset, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=asset,
                contype=0, conaffinity=0, mass=0, group=2, rgba=rgba,
                **({"material": kwargs["material"]} if texture else {}))
        common = dict(friction=obj["friction"], condim=4, rgba=[.1, .9, .4, .25], group=3)
        if obj["hull"] is not None:
            hull = obj["hull"]
            spec.add_mesh(name=prefix + "_hull", uservert=hull.vertices.ravel(), userface=hull.faces.ravel())
            body.add_geom(name=prefix + "_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=prefix + "_hull", mass=obj["mass"], **common)
        else:
            volumes = np.array([np.prod(size) for _, size in obj["boxes"]])
            for index, ((center, size), fraction) in enumerate(zip(obj["boxes"], volumes / volumes.sum())):
                body.add_geom(name=f"{prefix}_collision_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=center, size=size, mass=obj["mass"] * fraction, **common)
        print(f"Imported {obj['name']}: {obj['provenance']}, {obj['mode']} collision, "
              f"{'fixed' if obj['fixed'] else str(obj['mass']) + ' kg (assumed)'}", flush=True)
    return objects


def _read(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir() or path.name == "object.json":
        folder = path if path.is_dir() else path.parent
        config = object_assets.load(folder)
        place = config["placement"]
        return path, {"version": "object", "objects": [{
            "id": config["id"], "asset": folder, "position_m": place["position_m"],
            "yaw_degrees": place.get("yaw_degrees", 0), "fixed": place.get("fixed", False)}]}
    config = json.loads(path.read_text(encoding="utf-8"))
    return path, config


def _placements(path):
    """Validated version-2 / single-object placements: list of dicts."""
    path, config = _read(path)
    objects = config.get("objects", [])
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"Scene requires 1-{MAX_OBJECTS} objects")
    result, names = [], set()
    for obj in objects:
        name = identifier(obj["id"])
        if name in names:
            raise ValueError(f"Duplicate object id: {name}")
        names.add(name)
        asset = obj["asset"]
        if not isinstance(asset, Path):
            # A bare id means assets/objects/<id>; anything else stays inside
            # the scene's folder, like the version-1 bundle paths.
            if isinstance(asset, str) and object_assets.ID_PATTERN.fullmatch(asset) \
                    and not (path.parent / asset).exists():
                asset = object_assets.resolve(asset)
            else:
                if not isinstance(asset, str) or not asset or Path(asset).is_absolute():
                    raise ValueError("asset must be an object id or a relative folder inside the scene")
                asset = (path.parent / asset).resolve()
                if not asset.is_relative_to(path.parent.resolve()):
                    raise ValueError(f"Missing asset or path outside scene bundle: {obj['asset']}")
        object_assets.load(asset)
        yaw = float(obj.get("yaw_degrees", 0))
        fixed = obj.get("fixed", False)
        if not np.isfinite(yaw) or not isinstance(fixed, bool):
            raise ValueError("yaw_degrees must be finite and fixed true/false")
        result.append({"name": PREFIX + name, "asset": asset,
                       "position": vector(obj["position_m"], 3, "position_m"),
                       "yaw": yaw, "fixed": fixed})
    return result


def _version(path):
    path, config = _read(path)
    version = config.get("version")
    if version not in (1, 2, "object"):
        raise ValueError("Unsupported scene version (expected 1 or 2, or an object folder)")
    return version


def object_names(config_path):
    """Body names (photo_<id>) the scene adds, without building it."""
    if _version(config_path) == 1:
        _, config = _read(config_path)
        objects = config.get("objects", [])
        if not isinstance(objects, list) or not objects:
            raise ValueError("Scene requires at least one object")
        return [PREFIX + identifier(obj["id"]) for obj in objects]
    return [place["name"] for place in _placements(config_path)]


def scene_digest(config_path):
    """sha256 over the scene file and every object file it uses."""
    path = Path(config_path).expanduser().resolve()
    sha = hashlib.sha256()
    if _version(path) == 1:
        root = path.parent
        sha.update(path.read_bytes())
        for file in sorted(p for p in root.rglob("*") if p.is_file() and p != path):
            sha.update(file.relative_to(root).as_posix().encode() + b"\0" + file.read_bytes())
        return sha.hexdigest()
    if path.is_file():
        sha.update(path.read_bytes())
    for place in _placements(path):
        sha.update(place["name"].encode() + b"\0" + object_assets.digest(place["asset"]).encode())
    return sha.hexdigest()


def scene_summary(config_path, project_root=None):
    """What an episode stores: path (project-relative when inside it), sha256, objects."""
    path = Path(config_path).expanduser().resolve()
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    shown = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    summary = {"path": shown, "sha256": scene_digest(path), "objects": object_names(path)}
    if _version(path) != 1:
        summary["assets"] = {place["name"]: object_assets.summary(place["asset"])
                             for place in _placements(path)}
    return summary


def add_photo_scene(spec, config_path):
    """Add the scene's objects to `spec`; visual meshes have no contacts or mass."""
    if _version(config_path) == 1:
        return _add_legacy_scene(spec, config_path)
    placements = _placements(config_path)
    for place in placements:
        config = object_assets.load(place["asset"])
        center = place["position"] + [0, 0, config["size_m"][2] / 2]
        object_assets.add_body(spec, place["asset"], place["name"], center, place["yaw"], place["fixed"])
        collision = config["collision"]
        pieces = len(collision.get("hulls") or collision.get("boxes") or [])
        method = collision.get("method", collision["mode"])
        mass = "fixed" if place["fixed"] else \
            f"{config['mass_kg']} kg ({config.get('values', {}).get('mass', 'assumed')})"
        print(f"Imported {place['name'][len(PREFIX):]}: {config['source']['kind']}, "
              f"{method} collision ({pieces} pieces), {mass}", flush=True)
    return placements
