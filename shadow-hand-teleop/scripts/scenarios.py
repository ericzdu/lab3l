"""Scenario files: new scored tasks as data, no Python edits.

A scenario (assets/scenarios/<name>.json) places objects on the table and
lists the steps to score:

    {"schema": "lab3l.scenario", "version": 1, "name": "fruit_to_plate",
     "description": "...",
     "objects": [
       {"id": "apple", "asset": "apple", "position_m": [0.24, 0.04], "yaw_degrees": 0},
       {"id": "plate", "prop": "real_plate"},
       {"id": "mug", "asset": "mug", "position_m": [0.30, 0.20], "extra": true}],
     "steps": [{"object": "apple", "target": "plate", "hand": "left", "upright": false}],
     "scripted": {"grasp_assist": true, "grasp": {"apple": {"offset": [0.10, -0.03, -0.02]}},
                  "place_offset": {"apple": [0, -0.03]}}}

* An object is either an `asset` (a folder under assets/objects) placed at
  `position_m` = [x, y] on the worktop (its bottom rests on the table) and
  `yaw_degrees`, or a built-in `prop` from workspace_scene.PROPS at its usual
  spot. Every object listed is put on the table; ones no step uses are not
  scored, and `extra: true` says so explicitly (a step may not use it). The ids "cup"
  and "tray" name the workspace's own cup and tray.
* Each step moves `object` onto `target` with one hand. Steps are done in any
  order and scored on the final state, like set_the_table.
* `scripted` is read only by scripted_prop_tasks.py (grasp tweaks, assist).

register() loads every file into workspace_scene: each asset object becomes a
PROPS entry named <scenario>_<id>, and the scenario becomes a TASKS entry
named <scenario> (plus one entry per step, <scenario>_<object>, when there are
several). The sim, labels, recorder, replay and Studio all look tasks up by
name, so they work with scenarios unchanged. Episodes store the file's sha256
(metadata.scene.scenario); replay warns if it changed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = PROJECT_ROOT / "assets" / "scenarios"
SCHEMA = "lab3l.scenario"
SCHEMA_VERSION = 1
NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,23}")
ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,15}")
FIXTURES = ("cup", "tray")
# Worktop (human_arms.build_human_model): x .16-1.08, y -.58-.58.
WORKTOP_X = (.16, 1.08)
WORKTOP_Y = (-.58, .58)
# Footprints must be at least this far apart (and from the cup and tray).
CLEARANCE = .005
CUP_FOOTPRINT_RADIUS = .065   # wall radius plus the handle

# What register() added, so a re-register can take it back out.
_REGISTERED = {"props": [], "tasks": []}
LOADED = {}    # scenario name -> loaded scenario dict
ERRORS = {}    # file stem -> error message, for files that failed to load


# -- footprints and overlap ------------------------------------------------------
def _polygon(shape, size, center, yaw_degrees=0., grow=0.):
    """Top-view outline: (N, 2) points, counter-clockwise."""
    center = np.asarray(center, dtype=float)[:2]
    if shape == "round":
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return center + (float(size) + grow) * np.c_[np.cos(angles), np.sin(angles)]
    hx, hy = np.asarray(size, dtype=float) + grow
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    angle = np.deg2rad(yaw_degrees)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return center + corners @ turn.T


def _overlap(a, b):
    """True if two convex polygons intersect (separating-axis test)."""
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for normal in np.c_[-edges[:, 1], edges[:, 0]]:
            pa, pb = a @ normal, b @ normal
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def footprint_polygon(name, grow=0.):
    """Outline of a placed object: "cup", "tray", or a PROPS name (with its yaw)."""
    import workspace_scene as ws
    if name == "cup":
        return _polygon("round", CUP_FOOTPRINT_RADIUS, ws.CUP_START, grow=grow)
    if name == "tray":
        return _polygon("box", ws.TRAY_HALF_EXTENTS, ws.TRAY_CENTER, grow=grow)
    spec = ws.PROPS[name]
    shape, size = ws.prop_footprint(name)
    return _polygon(shape, size, spec["xy"], spec.get("yaw", 0.), grow=grow)


def overlaps(names):
    """Pairs of placed objects (plus the cup and tray) whose footprints overlap."""
    names = [*FIXTURES, *(n for n in names if n not in FIXTURES)]
    outlines = {name: footprint_polygon(name, CLEARANCE / 2) for name in names}
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if not (a in FIXTURES and b in FIXTURES) and _overlap(outlines[a], outlines[b])]


# -- loading ---------------------------------------------------------------------
def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def load(path):
    """Read and check one scenario file against the current PROPS/TASKS.

    Returns {"name", "path", "sha256", "description", "objects": {id: {...}},
    "steps": [...], "scripted": {...}}; raises ValueError naming the bad field.
    """
    import object_assets
    import workspace_scene as ws
    path = Path(path)
    raw = path.read_bytes()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from None
    if not isinstance(config, dict) or config.get("schema") != SCHEMA \
            or config.get("version") != SCHEMA_VERSION:
        raise ValueError(f'needs "schema": "{SCHEMA}", "version": {SCHEMA_VERSION}')
    name = config.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError("name must be 1-24 lowercase letters, digits or '_', starting with a letter")
    if name != path.stem:
        raise ValueError(f'name "{name}" must match the file name ({path.stem}.json)')
    if name in ws.TASKS:
        raise ValueError(f'name "{name}" is already a task; choose another')
    unknown = set(config) - {"schema", "version", "name", "description", "objects", "steps", "scripted"}
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(unknown)}")

    objects = {}
    entries = config.get("objects")
    if not isinstance(entries, list) or not entries:
        raise ValueError("objects must be a non-empty list")
    for index, entry in enumerate(entries):
        where = f"objects[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object")
        object_id = entry.get("id")
        if not isinstance(object_id, str) or not ID_PATTERN.fullmatch(object_id):
            raise ValueError(f"{where}.id must be 1-16 lowercase letters, digits or '_'")
        if object_id in FIXTURES:
            raise ValueError(f'{where}.id "{object_id}" is reserved for the workspace {object_id}')
        if object_id in objects:
            raise ValueError(f'{where}.id "{object_id}" is used twice')
        extra = entry.get("extra", False)
        if not isinstance(extra, bool):
            raise ValueError(f"{where}.extra must be true or false")
        if ("asset" in entry) == ("prop" in entry):
            raise ValueError(f'{where} needs exactly one of "asset" or "prop"')
        if "prop" in entry:
            unknown = set(entry) - {"id", "prop", "extra"}
            if unknown:
                raise ValueError(f"{where}: {sorted(unknown)} not allowed with prop (it keeps its spot)")
            prop = entry["prop"]
            if prop not in ws.PROPS or "scenario" in ws.PROPS[prop]:
                raise ValueError(f'{where}.prop "{prop}" is not a built-in prop '
                                 f"({', '.join(n for n in ws.PROPS if 'scenario' not in ws.PROPS[n])})")
            objects[object_id] = {"prop": prop, "extra": extra}
            continue
        unknown = set(entry) - {"id", "asset", "position_m", "yaw_degrees", "extra"}
        if unknown:
            raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
        asset = entry["asset"]
        if not isinstance(asset, str) or not object_assets.ID_PATTERN.fullmatch(asset):
            raise ValueError(f"{where}.asset must be an object id under assets/objects")
        try:
            object_assets.load(asset)
        except (ValueError, OSError, KeyError) as exc:
            raise ValueError(f'{where}.asset "{asset}": {exc}') from None
        position = entry.get("position_m")
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"{where}.position_m must be [x, y] in metres")
        xy = tuple(_number(v, f"{where}.position_m") for v in position)
        yaw = _number(entry.get("yaw_degrees", 0), f"{where}.yaw_degrees")
        prop = f"{name}_{object_id}"
        if prop in ws.PROPS:
            raise ValueError(f'{where}: prop name "{prop}" already exists')
        objects[object_id] = {"prop": prop, "extra": extra,
                              "spec": {"type": "asset", "asset": asset, "xy": xy, "yaw": yaw,
                                       "scenario": name}}

    steps = []
    entries = config.get("steps")
    if not isinstance(entries, list) or not entries:
        raise ValueError("steps must be a non-empty list")
    moved = set()
    for index, entry in enumerate(entries):
        where = f"steps[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object")
        unknown = set(entry) - {"object", "target", "hand", "upright"}
        if unknown:
            raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
        obj, target = entry.get("object"), entry.get("target")
        for label, value, allowed in (("object", obj, ("cup",)), ("target", target, ("tray",))):
            if value not in objects and value not in allowed:
                raise ValueError(f'{where}.{label} "{value}" is not one of this scenario\'s objects '
                                 f"({', '.join([*objects, *allowed])})")
            if value in objects and objects[value]["extra"]:
                raise ValueError(f'{where}.{label} "{value}" is marked extra (not scored)')
        if obj == target:
            raise ValueError(f"{where}: object and target are the same")
        if obj in moved:
            raise ValueError(f'{where}: "{obj}" is already moved by an earlier step')
        moved.add(obj)
        hand = entry.get("hand")
        if hand not in ("left", "right"):
            raise ValueError(f'{where}.hand must be "left" or "right"')
        upright = entry.get("upright", False)
        if not isinstance(upright, bool):
            raise ValueError(f"{where}.upright must be true or false")
        prop = lambda key: key if key in FIXTURES else objects[key]["prop"]
        steps.append({"id": obj, "object": prop(obj), "target": prop(target), "hand": hand,
                      "upright": upright})

    scripted = config.get("scripted", {})
    if not isinstance(scripted, dict) or set(scripted) - {"grasp_assist", "grasp", "place_offset"}:
        raise ValueError('scripted may only hold "grasp_assist", "grasp" and "place_offset"')
    if not isinstance(scripted.get("grasp_assist", True), bool):
        raise ValueError("scripted.grasp_assist must be true or false")
    def numbers(value, length, label):
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"{label} needs {length} numbers")
        return [_number(v, label) for v in value]

    for key in ("grasp", "place_offset"):
        table = scripted.get(key, {})
        if not isinstance(table, dict):
            raise ValueError(f"scripted.{key} must map object ids to values")
        for object_id, value in table.items():
            label = f"scripted.{key}.{object_id}"
            if object_id not in objects:
                raise ValueError(f'{label}: "{object_id}" is not one of this scenario\'s objects')
            if key == "place_offset":
                numbers(value, 2, label)
                continue
            if not isinstance(value, dict) or not value or set(value) - {"offset", "at"}:
                raise ValueError(f'{label} may hold "offset" [x, y, z] and "at" [x, y]')
            for field, length in (("offset", 3), ("at", 2)):
                if field in value:
                    numbers(value[field], length, f"{label}.{field}")

    description = config.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be text")
    scenario = {"name": name, "path": path.resolve(), "sha256": hashlib.sha256(raw).hexdigest(),
                "description": description, "objects": objects, "steps": steps,
                "scripted": scripted}
    _check_layout(scenario)
    return scenario


def _check_layout(scenario):
    """Every asset object on the worktop; no footprints overlapping (with the cup and tray)."""
    import workspace_scene as ws
    added = {o["prop"]: o["spec"] for o in scenario["objects"].values() if "spec" in o}
    ws.PROPS.update(added)   # footprint_polygon reads PROPS; taken out again below
    try:
        for object_id, entry in scenario["objects"].items():
            if "spec" not in entry:
                continue
            outline = footprint_polygon(entry["prop"])
            low, high = outline.min(axis=0), outline.max(axis=0)
            if low[0] < WORKTOP_X[0] or high[0] > WORKTOP_X[1] \
                    or low[1] < WORKTOP_Y[0] or high[1] > WORKTOP_Y[1]:
                raise ValueError(f'"{object_id}" at {list(entry["spec"]["xy"])} is not fully on the '
                                 f"worktop (x {WORKTOP_X[0]}-{WORKTOP_X[1]}, y {WORKTOP_Y[0]}-{WORKTOP_Y[1]})")
        props = [o["prop"] for o in scenario["objects"].values()]
        names = {o["prop"]: object_id for object_id, o in scenario["objects"].items()}
        clashes = overlaps(props)
        if clashes:
            shown = ", ".join(f"{names.get(a, a)} and {names.get(b, b)}" for a, b in clashes)
            raise ValueError(f"footprints overlap (or are closer than {CLEARANCE * 1000:.0f} mm): {shown}")
    finally:
        for prop in added:
            ws.PROPS.pop(prop, None)


# -- registering into workspace_scene -------------------------------------------------
def _add(scenario):
    import workspace_scene as ws
    name = scenario["name"]
    grasp = scenario["scripted"].get("grasp", {})
    for object_id, entry in scenario["objects"].items():
        if "spec" in entry:
            spec = dict(entry["spec"])
            if object_id in grasp:
                spec["scripted_grasp"] = grasp[object_id]
            ws.PROPS[entry["prop"]] = spec
            _REGISTERED["props"].append(entry["prop"])
    # Everything listed goes on the table; what no step moves or targets is unscored.
    used = {name for step in scenario["steps"] for name in (step["object"], step["target"])}
    extras = [o["prop"] for o in scenario["objects"].values() if o["prop"] not in used]
    offsets = {scenario["objects"][k]["prop"]: tuple(v)
               for k, v in scenario["scripted"].get("place_offset", {}).items()}
    common = {"scenario": name, "extras": extras,
              "grasp_assist": scenario["scripted"].get("grasp_assist", True),
              "place_offsets": offsets}
    single = lambda step: {"object": step["object"], "target": step["target"],
                           "upright": step["upright"], "hand": step["hand"]}
    if len(scenario["steps"]) == 1:
        ws.TASKS[name] = {**single(scenario["steps"][0]), **common}
        _REGISTERED["tasks"].append(name)
        return
    names = []
    for step in scenario["steps"]:
        step_name = f"{name}_{step['id']}"
        if step_name in ws.TASKS:
            raise ValueError(f'step task name "{step_name}" already exists')
        names.append(step_name)
    for step, step_name in zip(scenario["steps"], names):
        ws.TASKS[step_name] = {**single(step), "scenario": name, "part_of": name}
        _REGISTERED["tasks"].append(step_name)
    ws.TASKS[name] = {"steps": names, **common}
    _REGISTERED["tasks"].append(name)


def register(directory=None, quiet=False):
    """(Re)load every scenario file in `directory` (default assets/scenarios).

    Takes out whatever an earlier call added first, so it can be called again
    to pick up new or edited files. A file that fails to load is skipped and
    its error kept in ERRORS (and printed unless quiet). Returns the names loaded.
    """
    import workspace_scene as ws
    for prop in _REGISTERED["props"]:
        ws.PROPS.pop(prop, None)
    for task in _REGISTERED["tasks"]:
        ws.TASKS.pop(task, None)
    _REGISTERED["props"].clear()
    _REGISTERED["tasks"].clear()
    LOADED.clear()
    ERRORS.clear()
    folder = Path(directory) if directory is not None else SCENARIOS_DIR
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            scenario = load(path)
            _add(scenario)
        except (ValueError, OSError) as exc:
            ERRORS[path.stem] = str(exc)
            if not quiet:
                print(f"WARNING: scenario {path.name} skipped: {exc}", file=sys.stderr, flush=True)
            continue
        LOADED[scenario["name"]] = scenario
    return list(LOADED)


def scenario_of(task):
    """The loaded scenario a task belongs to, or None for a built-in task."""
    import workspace_scene as ws
    return LOADED.get((ws.TASKS.get(task) or {}).get("scenario"))


def summary(task):
    """What an episode stores about its scenario (metadata.scene.scenario), or None."""
    scenario = scenario_of(task)
    if scenario is None:
        return None
    path = scenario["path"]
    shown = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
    return {"name": scenario["name"], "path": shown, "sha256": scenario["sha256"]}
