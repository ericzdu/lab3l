"""Preview sheet for an object: front, side and top views on a 1 cm grid, as one PNG.

Used by prepare_object.py --preview-only (a mesh scaled and turned in memory,
seconds, nothing written but the PNG) and --preview (the finished folder, plus a
fourth panel of its collision pieces, one colour each). Views follow the sim's
cameras: "front" is the operator's view (looking along +x), "side" looks from
the robot's right (+y), "top" looks down with +x up and +y to the left.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

VIEWS = (("front", 0, -20), ("side (from the right)", 90, -20), ("top", 0, -89.9))
COLLISION_VIEW = ("collision pieces", 30, -30)
VISUAL_GROUP, COLLISION_GROUP, FLOOR_GROUP = 0, 1, 2
PIECE_COLORS = ((.90, .30, .25), (.25, .60, .90), (.30, .80, .40), (.95, .75, .20),
                (.65, .40, .85), (.20, .80, .80), (.95, .50, .70), (.55, .55, .55))
PANEL_PX = 480
CAPTION_PX = 56
GRID_SQUARE_M = .01


def render_sheet(visuals, size_m, path, caption, hulls=(), boxes=()):
    """Render `visuals` [(mesh path, texture path or None, rgba)] to a PNG at `path`.

    Meshes are metres, +Z up, bottom centre at the origin (object.json frame).
    hulls: collision mesh paths; boxes: [(centre, half size)]. Either adds a
    fourth panel showing only the collision model, one colour per piece.
    """
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    size_m = np.asarray(size_m, dtype=float)
    reach = float(max(size_m))
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = PANEL_PX
    spec.visual.global_.offheight = PANEL_PX
    spec.visual.headlight.ambient = [.45, .45, .45]
    spec.visual.headlight.diffuse = [.55, .55, .55]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[1, 1, 1], rgb2=[.85, .87, .90], width=64, height=64)
    # Checker texture: 2x2 squares per tile, so a tile of 2 grid squares.
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.94, .94, .94], rgb2=[.80, .80, .80], width=64, height=64)
    slots = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"
    spec.add_material(name="grid", textures=slots, texuniform=True,
                      texrepeat=[1 / (2 * GRID_SQUARE_M)] * 2)
    floor = max(.3, 3 * reach)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[floor, floor, .01], material="grid",
                            group=FLOOR_GROUP)
    no_contact = dict(contype=0, conaffinity=0)
    for index, (mesh, texture, rgba) in enumerate(visuals):
        name = f"visual_{index}"
        spec.add_mesh(name=name, file=str(mesh), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
        kwargs = {"rgba": list(rgba)}
        if texture is not None:
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture))
            slots = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
            slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = name
            spec.add_material(name=name, textures=slots, rgba=[1, 1, 1, 1])
            kwargs = {"material": name, "rgba": [1, 1, 1, 1]}
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, group=VISUAL_GROUP,
                                **no_contact, **kwargs)
    color = lambda i: [*PIECE_COLORS[i % len(PIECE_COLORS)], 1]
    for index, mesh in enumerate(hulls):
        spec.add_mesh(name=f"hull_{index}", file=str(mesh))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"hull_{index}",
                                rgba=color(index), group=COLLISION_GROUP, **no_contact)
    for index, (center, half) in enumerate(boxes):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half),
                                rgba=color(index), group=COLLISION_GROUP, **no_contact)
    views = [(label, a, e, VISUAL_GROUP) for label, a, e in VIEWS]
    if len(hulls) or len(boxes):
        views.append((*COLLISION_VIEW, COLLISION_GROUP))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sheet = Image.new("RGB", (len(views) * PANEL_PX, PANEL_PX + CAPTION_PX), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    with mujoco.Renderer(model, height=PANEL_PX, width=PANEL_PX) as renderer:
        for index, (label, azimuth, elevation, group) in enumerate(views):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, size_m[2] / 2]
            camera.distance = 1.7 * reach + .03
            camera.azimuth, camera.elevation = azimuth, elevation
            option = mujoco.MjvOption()
            option.geomgroup[:] = 0
            option.geomgroup[group] = option.geomgroup[FLOOR_GROUP] = 1
            renderer.update_scene(data, camera, option)
            sheet.paste(Image.fromarray(renderer.render()), (index * PANEL_PX, 0))
            left = index * PANEL_PX
            draw.rectangle(draw.textbbox((left + 10, 8), label, font=font), fill="white")
            draw.text((left + 10, 8), label, fill="black", font=font)
    draw.text((10, PANEL_PX + 8), caption, fill="black", font=font)
    draw.text((10, PANEL_PX + 30), f"Grid squares {GRID_SQUARE_M * 100:.0f} cm.", fill="black", font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return Path(path)


def size_text(size_m):
    x, y, z = (100 * np.asarray(size_m, dtype=float)).round(1)
    return f"{x} x {y} x {z} cm (x, y, z)"


def render_source(source, path, *, up_axis, height_m=None, width_m=None, scale=None, rotate_z=0.,
                  align_footprint=False, texture_size=1024):
    """Scale and turn a mesh in memory (no CoACD, no folder) and render it. Returns the size (m)."""
    from prepare_object import normalize_parts, part_appearance, triangle_parts, write_visuals
    parts, _ = normalize_parts(triangle_parts(source), height_m, up_axis, width_m=width_m,
                               scale=scale, rotate_z=rotate_z, align_footprint=align_footprint)
    points = np.concatenate([part.vertices for part in parts])
    size = points.max(axis=0) - points.min(axis=0)
    appearance = [part_appearance(part, texture_size) for part in parts]
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        records = write_visuals(folder, parts, appearance)
        visuals = [(folder / r["mesh"], folder / r["texture"] if "texture" in r else None, r["rgba"])
                   for r in records]
        caption = (f"{Path(source).name}: {size_text(size)} · up-axis {up_axis}, rotate-z {rotate_z:g}"
                   " · preview only, not settled")
        render_sheet(visuals, size, path, caption)
    return size


def render_folder(folder, path):
    """Render a prepared object folder, with its collision model in green."""
    import object_assets
    config = object_assets.load(folder)
    visuals = [(p["_mesh"], p["_texture"], p.get("rgba", [.7, .7, .7, 1])) for p in config["visual"]]
    collision = config["collision"]
    hulls = [h["_mesh"] for h in collision.get("hulls", [])]
    boxes = [(b["position_m"], b["half_size_m"]) for b in collision.get("boxes", [])]
    prep = config.get("preparation", {})
    pieces = len(hulls) or len(boxes)
    caption = (f"{config['id']}: {size_text(config['size_m'])} · {config['mass_kg'] * 1000:g} g · "
               f"{pieces} collision piece{'s' if pieces != 1 else ''}"
               + (f" · settled, tilted {prep['settle_tilt_degrees']:g} deg" if prep.get("settled") else ""))
    return render_sheet(visuals, config["size_m"], path, caption, hulls=hulls, boxes=boxes)
