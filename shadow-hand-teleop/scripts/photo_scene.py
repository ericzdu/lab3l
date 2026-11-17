"""Local triangle-mesh -> MuJoCo adapter, NOT a SAM 3D inference implementation.

Optional trimesh dependency is loaded only when this adapter is used. Scene
bundles use metres, kilograms, Z-up, and a bottom-centred object origin.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

MAX_FACES = 100_000
MAX_PARTS = 64
MAX_OBJECTS = 32


def mesh_library():
    try:
        import trimesh
    except ImportError as exc:
        raise ValueError("Install photo-scene dependencies: .venv/bin/python -m pip install -r requirements-photo-scene.txt") from exc
    return trimesh


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


def triangle_parts(source):
    """Bake GLB scene-node transforms; refuse point clouds / Gaussian PLYs."""
    tm = mesh_library()
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() not in (".glb", ".gltf", ".obj", ".stl", ".ply"):
        raise ValueError("Use a GLB/GLTF/OBJ/STL or triangle-mesh PLY export, not an image")
    if not source.is_file() or source.stat().st_size > 150_000_000:
        raise ValueError("Input must be an existing local mesh smaller than 150 MB")
    scene = tm.load_scene(source, process=False, allow_remote=False)
    parts = []
    total_faces = 0
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        if not isinstance(mesh, tm.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("Point clouds / Gaussian splats are not collision meshes. Export the triangle mesh (prefer GLB).")
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Mesh contains non-finite vertices")
        if len(mesh.vertices) < 4 or np.any(mesh.extents < 1e-8):
            raise ValueError("Mesh part must have nonzero extent in all three axes")
        total_faces += len(mesh.faces)
        parts.append(mesh)
    if not parts:
        raise ValueError("No triangle-mesh geometry found")
    if total_faces > MAX_FACES or len(parts) > MAX_PARTS:
        raise ValueError(f"Mesh too large: simplify below {MAX_FACES} faces / {MAX_PARTS} parts first")
    return parts


def normalize_parts(parts, height_m, up_axis):
    """Uniform scale preserves aspect ratio; no metric scale is inferred."""
    height_m = positive(height_m, "height_m")
    if height_m > 3:
        raise ValueError("This tabletop prototype supports objects up to 3 m high")
    tm = mesh_library()
    transforms = {
        "z": np.eye(4),
        "y": tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
        "x": tm.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]),
    }
    if up_axis not in transforms:
        raise ValueError("up_axis must be x, y or z")
    parts = [part.copy() for part in parts]
    for part in parts:
        part.apply_transform(transforms[up_axis])
    bounds = np.array([part.bounds for part in parts])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    scale = height_m / positive(high[2] - low[2], "input height")
    origin = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    for part in parts:
        part.vertices = (part.vertices - origin) * scale
    return parts, float(scale)


def prepare_bundle(source, destination, *, object_id, height_m, up_axis,
                   position=(.55, .28, .60), mass_kg=.15, fixed=False,
                   collision="convex", provenance="external_mesh"):
    """Create a NEW portable bundle; never overwrite an existing directory."""
    identifier(object_id)
    position = vector(position, 3, "position_m")
    mass_kg = positive(mass_kg, "mass_kg")
    if collision not in ("box", "convex"):
        raise ValueError("Converter collision must be box or convex")
    if provenance not in ("external_mesh", "sam3d_export", "synthetic_test"):
        raise ValueError("Unknown provenance kind")
    source = Path(source).expanduser().resolve()
    parts, scale = normalize_parts(triangle_parts(source), height_m, up_axis)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists; choose a new folder: {destination}")
    records, notes = [], []
    # Validate and prepare in memory before creating the output directory.
    for index, mesh in enumerate(parts):
        rgba = [.65, .55, .35, 1.]
        texture = None
        visual = mesh.visual
        if visual.kind == "texture":
            material = visual.material
            simple = material.to_simple() if hasattr(material, "to_simple") else material
            color = getattr(simple, "diffuse", None)
            if color is not None:
                rgba = (np.asarray(color, dtype=float) / 255).tolist()
            image = getattr(simple, "image", None)
            if image is not None and getattr(visual, "uv", None) is not None:
                uv = np.asarray(visual.uv)
                if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
                    raise ValueError("Texture requires finite per-vertex UVs")
                texture = image.convert("RGB")
                if max(texture.size) > 2048:
                    texture.thumbnail((2048, 2048))
        elif visual.kind in ("vertex", "face"):
            rgba = (np.mean(visual.face_colors, axis=0) / 255).tolist()
            notes.append("Vertex/face colors reduced to one mean color per mesh part.")
        records.append((mesh, rgba, texture, f"visual_{index}"))
    destination.mkdir(parents=True)
    part_records = []
    for mesh, rgba, texture, stem in records:
        # Keep vertex/UV indices aligned without external MTL references. The
        # texture is loaded explicitly from JSON, never fetched by OBJ/MTL.
        lines = ["# Prepared local mesh; units: metres; up: Z"]
        lines += ["v " + " ".join(f"{value:.12g}" for value in vertex) for vertex in mesh.vertices]
        if texture is not None:
            uv = np.asarray(mesh.visual.uv)
            lines += ["vt " + " ".join(f"{value:.12g}" for value in coord) for coord in uv]
            lines += ["f " + " ".join(f"{idx + 1}/{idx + 1}" for idx in face) for face in mesh.faces]
        else:
            lines += ["f " + " ".join(str(idx + 1) for idx in face) for face in mesh.faces]
        (destination / f"{stem}.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
        record = {"mesh": f"{stem}.obj", "rgba": rgba}
        if texture is not None:
            texture.save(destination / f"{stem}.png")
            record["texture"] = f"{stem}.png"
        part_records.append(record)
    all_vertices = np.concatenate([p.vertices for p in parts])
    config = {
        "version": 1,
        "description": "Mesh-import feasibility prototype. Not a calibrated reconstruction of a real environment.",
        "objects": [{
            "id": object_id,
            "provenance": {"kind": provenance, "source_name": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "inference_run_verified": False,
                "note": "Source kind is user-declared; this importer does not run or verify SAM 3D inference."},
            "parts": part_records, "position_m": position.tolist(), "yaw_degrees": 0,
            "size_m": np.ptp(all_vertices, axis=0).tolist(),
            "fixed": bool(fixed), "mass_kg": mass_kg, "friction": [1., .01, .001],
            "collision": {"mode": collision},
            "preparation": {"source_up_axis": up_axis, "scale_to_metres": scale,
                "height_source": "user-specified, not estimated from image",
                "physics_source": "manual assumptions, not inferred from image",
                "notes": sorted(set(notes)) + ["Base color/UV texture only; PBR lighting properties are not preserved."]}
        }]
    }
    output = destination / "scene.json"
    output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def load_bundle(config_path):
    """Validate all assets before mutating a MuJoCo spec."""
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


def add_photo_scene(spec, config_path):
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
