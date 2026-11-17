"""Prepare a 3D mesh (SAM 3D export, scan, CAD) as a simulation object folder.

    .venv/bin/python scripts/prepare_object.py apple.glb assets/objects/apple \\
        --id apple --up-axis y --height 0.072 --mass 0.068 --measured height mass \\
        --source-kind sam3d_export --footprint round

Output (a NEW folder; existing folders are never overwritten):
object.json, visual_<i>.obj (+ .png texture), collision/hull_*.obj and, with
--keep-source, the original file under source/. See docs/ADDING_OBJECTS.md.

The mesh becomes metres, Z-up, bottom centre at the origin. Exactly one of
--height / --width / --scale sets the size (uniform: aspect ratio is kept).
The collision model is a CoACD convex decomposition by default. Mass,
friction and size are NOT inferred from the mesh: they are your inputs, and
object.json records whether each was measured or assumed.

Needs the optional dependencies: pip install -r requirements-photo-scene.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from object_assets import ID_PATTERN, SCHEMA, SCHEMA_VERSION, SOURCE_KINDS, load

MAX_FACES = 100_000
MAX_PARTS = 64
MAX_INPUT_BYTES = 150_000_000
DEFAULT_FRICTION = (1., .005, .001)


def mesh_library():
    try:
        import trimesh
    except ImportError as exc:
        raise ValueError("Install the mesh tools: .venv/bin/python -m pip install "
                         "-r requirements-photo-scene.txt") from exc
    return trimesh


def positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def triangle_parts(source):
    """Triangle meshes with scene-node transforms baked; refuse point clouds / splats."""
    tm = mesh_library()
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() not in (".glb", ".gltf", ".obj", ".stl", ".ply"):
        raise ValueError("Use a GLB/GLTF/OBJ/STL or triangle-mesh PLY export, not an image")
    if not source.is_file() or source.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Input must be an existing local mesh smaller than 150 MB")
    scene = tm.load_scene(source, process=False, allow_remote=False)
    parts, total_faces = [], 0
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        if not isinstance(mesh, tm.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("Point clouds / Gaussian splats are not collision meshes. "
                             "Export the triangle mesh (prefer GLB).")
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


def footprint_angle(parts):
    """Yaw (degrees) that turns the mesh's minimum-area top-view rectangle to the axes."""
    from scipy.spatial import ConvexHull
    points = np.concatenate([part.vertices[:, :2] for part in parts])
    hull = points[ConvexHull(points).vertices]
    best = (np.inf, 0.)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        angle = np.arctan2(*(b - a)[::-1])
        c, s = np.cos(-angle), np.sin(-angle)
        rotated = hull @ np.array([[c, -s], [s, c]]).T
        area = np.prod(np.ptp(rotated, axis=0))
        if area < best[0] - 1e-12:
            best = (area, -angle)
    # Smallest equivalent turn: rectangles repeat every 90 degrees.
    return float((np.degrees(best[1]) + 45) % 90 - 45)


def normalize_parts(parts, height_m=None, up_axis="z", *, width_m=None, scale=None,
                    rotate_z=0., align_footprint=False):
    """Z-up, bottom centre at the origin, uniformly scaled; returns (parts, scale).

    Exactly one of height_m (Z extent), width_m (largest horizontal extent)
    or scale (multiplier, e.g. 1 for a mesh already in metres) sets the size.
    rotate_z (degrees) turns the mesh about +Z after the up-axis change;
    align_footprint adds the turn that squares its top view to the axes (a
    box scanned at 45 degrees), so width and footprint measure the real sides.
    """
    if sum(value is not None for value in (height_m, width_m, scale)) != 1:
        raise ValueError("Give exactly one of height, width or scale")
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
    yaw = float(rotate_z) + (footprint_angle(parts) if align_footprint else 0.)
    if not np.isfinite(yaw):
        raise ValueError("rotate_z must be finite")
    if yaw:
        turn = tm.transformations.rotation_matrix(np.deg2rad(yaw), [0, 0, 1])
        for part in parts:
            part.apply_transform(turn)
    bounds = np.array([part.bounds for part in parts])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    if height_m is not None:
        factor = positive(height_m, "height_m") / positive(high[2] - low[2], "input height")
    elif width_m is not None:
        factor = positive(width_m, "width_m") / positive(max(high[:2] - low[:2]), "input width")
    else:
        factor = positive(scale, "scale")
    if (high[2] - low[2]) * factor > 3:
        raise ValueError("Objects up to 3 m high are supported; check the units")
    origin = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    for part in parts:
        part.vertices = (part.vertices - origin) * factor
    return parts, float(factor)


def part_appearance(mesh, texture_size):
    """(rgba, PIL texture or None, note or None) for one mesh part."""
    rgba, texture, note = [.65, .55, .35, 1.], None, None
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
            if max(texture.size) > texture_size:
                texture.thumbnail((texture_size, texture_size))
    elif visual.kind in ("vertex", "face"):
        rgba = (np.mean(visual.face_colors, axis=0) / 255).tolist()
        note = "Vertex/face colors reduced to one mean color per mesh part."
    return rgba, texture, note


def obj_text(vertices, faces, uv=None):
    """Minimal OBJ (no MTL); a texture is attached from object.json, never by OBJ."""
    lines = ["# lab3l object mesh; units: metres; up: +Z; origin: bottom centre"]
    lines += ["v " + " ".join(f"{value:.9g}" for value in vertex) for vertex in vertices]
    if uv is not None:
        lines += ["vt " + " ".join(f"{value:.9g}" for value in coord) for coord in uv]
        lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in face) for face in faces]
    else:
        lines += ["f " + " ".join(str(i + 1) for i in face) for face in faces]
    return "\n".join(lines) + "\n"


def collision_hulls(parts, method, max_hulls, threshold, max_hull_vertices, seed=0):
    """Convex pieces [(vertices, faces, volume)] for the collision model."""
    tm = mesh_library()
    combined = tm.util.concatenate(parts)
    if method == "convex":
        hulls = [combined.convex_hull]
    elif method == "coacd":
        try:
            import coacd
        except ImportError as exc:
            raise ValueError("CoACD is missing: pip install -r requirements-photo-scene.txt") from exc
        coacd.set_log_level("error")
        merged = tm.Trimesh(combined.vertices, combined.faces, process=True)
        pieces = coacd.run_coacd(coacd.Mesh(merged.vertices, merged.faces), threshold=threshold,
                                 max_convex_hull=max_hulls, max_ch_vertex=max_hull_vertices,
                                 preprocess_mode="auto", seed=seed)
        hulls = [tm.Trimesh(v, f).convex_hull for v, f in pieces]
    else:
        raise ValueError("collision must be coacd or convex here (box is written as a primitive)")
    result = []
    for hull in hulls:
        volume = float(abs(hull.volume))
        if volume > 1e-12 and len(hull.vertices) >= 4:
            result.append((np.asarray(hull.vertices), np.asarray(hull.faces), volume))
    if not result:
        raise ValueError("Collision decomposition produced no solid pieces")
    return result


def settle_rotation(hulls, mass_kg, friction, seconds=3.):
    """3x3 rotation the object comes to rest in when dropped on a flat floor.

    Simulates the collision pieces alone (MuJoCo, 2 ms steps). A curved or
    tilted scan (a banana, a SAM 3D export at an angle) otherwise rocks into
    a different pose every time it is spawned.
    """
    import mujoco
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, .1], friction=list(friction))
    height = max(float(v[:, 2].max()) for v, _, _ in hulls)
    body = spec.worldbody.add_body(pos=[0, 0, .002])
    body.add_freejoint()
    volumes = np.array([volume for _, _, volume in hulls])
    for index, ((vertices, faces, _), share) in enumerate(zip(hulls, volumes / volumes.sum())):
        spec.add_mesh(name=f"hull_{index}", uservert=vertices.ravel(), userface=faces.ravel())
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"hull_{index}",
                      mass=mass_kg * share, friction=list(friction), condim=6)
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or data.qpos[2] < -height:
        raise ValueError("Settling failed: the collision model fell through the floor")
    return data.xmat[1].reshape(3, 3).copy()


def prepare(source, destination, *, object_id, up_axis, height_m=None, width_m=None, scale=None,
            mass_kg, rotate_z=0., align_footprint=False, friction=DEFAULT_FRICTION, condim=4, footprint="box", collision="coacd",
            max_hulls=16, coacd_threshold=.05, max_hull_vertices=64, texture_size=1024,
            position=(.55, .28, .60), yaw_degrees=0., fixed=False, source_kind="external_mesh",
            source_url=None, license=None, attribution=None, measured=(), keep_source=False,
            notes=(), settle=False):
    """Create a NEW object folder; return its object.json path."""
    if not ID_PATTERN.fullmatch(str(object_id)):
        raise ValueError("id must start with a letter and use letters, digits or '_' (max 48)")
    mass_kg = positive(mass_kg, "mass_kg")
    friction = [float(v) for v in friction]
    if len(friction) != 3 or not np.isfinite(friction).all() or min(friction) < 0:
        raise ValueError("friction needs 3 non-negative numbers")
    if condim not in (1, 3, 4, 6):
        raise ValueError("condim must be 1, 3, 4 or 6")
    if footprint not in ("box", "round"):
        raise ValueError("footprint must be box or round")
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source kind must be one of {SOURCE_KINDS}")
    unknown = set(measured) - {"height", "width", "mass", "friction"}
    if unknown:
        raise ValueError(f"--measured accepts height, width, mass, friction (not {sorted(unknown)})")
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position needs 3 finite numbers")
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists; choose a new folder: {destination}")

    # Everything is computed in memory before the folder is created.
    parts, factor = normalize_parts(triangle_parts(source), height_m, up_axis, width_m=width_m,
                                    scale=scale, rotate_z=rotate_z, align_footprint=align_footprint)
    appearance = [part_appearance(part, texture_size) for part in parts]
    hulls = None if collision == "box" else \
        collision_hulls(parts, collision, max_hulls, coacd_threshold, max_hull_vertices)
    tilt = 0.
    if settle:
        if hulls is None:
            raise ValueError("--settle needs a mesh collision model (coacd or convex), not box")
        rotation = settle_rotation(hulls, mass_kg, friction)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        turn = np.eye(4)
        turn[:3, :3] = rotation
        for part in parts:
            part.apply_transform(turn)
        hulls = [(v @ rotation.T, f, volume) for v, f, volume in hulls]
        # Back to bottom centre at the origin.
        points = np.concatenate([part.vertices for part in parts])
        low, high = points.min(axis=0), points.max(axis=0)
        shift = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
        for part in parts:
            part.vertices = part.vertices - shift
        hulls = [(v - shift, f, volume) for v, f, volume in hulls]
    vertices = np.concatenate([part.vertices for part in parts])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    size = high - low
    if footprint == "round":
        shape = {"shape": "round", "radius_m": float(np.max(np.linalg.norm(vertices[:, :2], axis=1)))}
    else:
        shape = {"shape": "box", "half_extents_m": (size[:2] / 2).tolist()}
    source_bytes = source.read_bytes()

    destination.mkdir(parents=True)
    if hulls is not None:
        (destination / "collision").mkdir()
    visual_records = []
    for index, (part, (rgba, texture, _)) in enumerate(zip(parts, appearance)):
        stem = f"visual_{index}"
        uv = np.asarray(part.visual.uv) if texture is not None else None
        (destination / f"{stem}.obj").write_text(obj_text(part.vertices, part.faces, uv), encoding="utf-8")
        record = {"mesh": f"{stem}.obj", "rgba": [round(float(v), 4) for v in rgba]}
        if texture is not None:
            texture.save(destination / f"{stem}.png", optimize=True)
            record["texture"] = f"{stem}.png"
        visual_records.append(record)
    if collision == "box":
        # A MuJoCo box primitive: exact box-box contacts (a box mesh is not).
        collision_record = {"mode": "boxes", "method": "box", "boxes": [
            {"position_m": ((low + high) / 2).tolist(), "half_size_m": (size / 2).tolist()}]}
        pieces = 1
    else:
        hull_records = []
        for index, (hull_vertices, hull_faces, volume) in enumerate(hulls):
            name = f"collision/hull_{index:02d}.obj"
            (destination / name).write_text(obj_text(hull_vertices, hull_faces), encoding="utf-8")
            hull_records.append({"mesh": name, "volume_m3": volume})
        collision_record = {"mode": "hulls", "method": collision, "hulls": hull_records}
        pieces = len(hull_records)
    kept = None
    if keep_source:
        (destination / "source").mkdir()
        kept = f"source/{source.name}"
        shutil.copy2(source, destination / kept)

    if {"height", "width"} & set(measured):
        size_origin = "measured"
    elif source_kind == "scanned_dataset" and scale == 1:
        size_origin = "dataset"          # a metric scan used at its true size
    elif source_kind == "scanned_dataset":
        size_origin = "scaled_placeholder"
    else:
        size_origin = "assumed"
    origin = lambda key: "measured" if key in measured else "assumed"
    config = {
        "schema": SCHEMA, "version": SCHEMA_VERSION, "id": object_id,
        "description": f"Prepared from {source.name}. Mesh frame: metres, +Z up, origin at bottom centre.",
        "size_m": [round(float(v), 6) for v in size],
        "mass_kg": mass_kg,
        "contact": {"friction": friction, "condim": condim},
        "footprint": shape,
        "visual": visual_records,
        "collision": collision_record,
        "placement": {"position_m": position.tolist(), "yaw_degrees": float(yaw_degrees),
                      "fixed": bool(fixed)},
        "values": {"size": size_origin, "mass": origin("mass"), "friction": origin("friction")},
        "source": {"kind": source_kind, "name": source.name,
                   "sha256": hashlib.sha256(source_bytes).hexdigest(),
                   "url": source_url, "license": license, "attribution": attribution,
                   "kept_file": kept},
        "preparation": {"up_axis": up_axis, "scale_to_metres": factor,
                        "rotate_z_degrees": float(rotate_z), "align_footprint": bool(align_footprint),
                        "size_from": "height" if height_m is not None else
                                     "width" if width_m is not None else "scale",
                        "collision": {"method": collision, "max_hulls": max_hulls,
                                      "threshold": coacd_threshold,
                                      "max_hull_vertices": max_hull_vertices,
                                      "pieces": pieces},
                        "texture_max_px": texture_size,
                        "settled": bool(settle), "settle_tilt_degrees": round(tilt, 2)},
        "notes": sorted({note for _, _, note in appearance if note} | set(notes)
                        | {"Base colour texture only; PBR lighting properties are not preserved.",
                           "Mass, friction and size are inputs, not inferred from the mesh."}),
    }
    path = destination / "object.json"
    path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load(destination)   # validate what was written
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="GLB/GLTF/OBJ/STL/mesh-PLY; NOT a photo or Gaussian splat")
    parser.add_argument("destination", help="New folder, e.g. assets/objects/<id>")
    parser.add_argument("--id", required=True, dest="object_id")
    parser.add_argument("--up-axis", choices=("x", "y", "z"), required=True,
                        help="The source mesh's up axis (GLB is usually y; check visually)")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--height", type=float, dest="height_m", help="Real height in METRES")
    size.add_argument("--width", type=float, dest="width_m",
                      help="Real largest horizontal size in METRES (e.g. a plate's diameter)")
    size.add_argument("--scale", type=float, help="Multiplier; 1 for a mesh already in metres")
    parser.add_argument("--mass", type=float, required=True, dest="mass_kg", help="Mass in kg")
    parser.add_argument("--rotate-z", type=float, default=0., help="Turn the mesh about +Z (degrees)")
    parser.add_argument("--align-footprint", action="store_true",
                        help="Square the top view to the axes first (e.g. a box scanned at 45 degrees)")
    parser.add_argument("--friction", type=float, nargs=3, default=DEFAULT_FRICTION,
                        metavar=("SLIDE", "SPIN", "ROLL"))
    parser.add_argument("--condim", type=int, default=4, choices=(1, 3, 4, 6),
                        help="Contact dimension: 4 adds torsional friction, 6 also rolling (round things)")
    parser.add_argument("--footprint", choices=("box", "round"), default="box",
                        help="Shape of the top/bottom for task scoring (plate: round)")
    parser.add_argument("--collision", choices=("coacd", "convex", "box"), default="coacd",
                        help="coacd: convex decomposition (default); convex: one hull; box: bounding box")
    parser.add_argument("--max-hulls", type=int, default=16)
    parser.add_argument("--coacd-threshold", type=float, default=.05,
                        help="Lower = closer fit, more pieces (0.01-1)")
    parser.add_argument("--max-hull-vertices", type=int, default=64)
    parser.add_argument("--texture-size", type=int, default=1024, help="Largest texture side in px")
    parser.add_argument("--position", type=float, nargs=3, default=(.55, .28, .60),
                        metavar=("X", "Y", "Z"), help="Default placement (bottom centre) for --scene-config")
    parser.add_argument("--yaw", type=float, default=0., dest="yaw_degrees")
    parser.add_argument("--fixed", action="store_true", help="Static fixture, not a movable object")
    parser.add_argument("--source-kind", choices=SOURCE_KINDS, default="external_mesh")
    parser.add_argument("--source-url")
    parser.add_argument("--license")
    parser.add_argument("--attribution")
    parser.add_argument("--measured", nargs="*", default=[],
                        help="Which inputs you measured on the real object: height width mass friction")
    parser.add_argument("--keep-source", action="store_true",
                        help="Copy the original file into source/ (recommended for SAM 3D GLBs)")
    parser.add_argument("--settle", action="store_true",
                        help="Turn the object into the pose it rests in on a table (recommended for "
                             "SAM 3D exports and curved objects)")
    parser.add_argument("--note", action="append", default=[], dest="notes")
    args = parser.parse_args(argv)
    try:
        path = prepare(**vars(args))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Cannot prepare object: {exc}\n")
    config = load(path.parent)
    print(f"Prepared {path}")
    print(f"Size {np.round(np.asarray(config['size_m']) * 1000).astype(int).tolist()} mm, "
          f"{config['preparation']['collision']['pieces']} collision pieces.")
    print(f'View it: zsh ./run_photo_scene.command "{path.parent}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
