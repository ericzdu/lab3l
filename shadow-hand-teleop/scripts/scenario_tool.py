"""List, check and create scenario files (assets/scenarios/*.json, see scenarios.py).

    .venv/bin/python scripts/scenario_tool.py list
    .venv/bin/python scripts/scenario_tool.py check fruit_to_plate --snapshot layout.png
    .venv/bin/python scripts/scenario_tool.py check fruit_to_plate --scripted
    .venv/bin/python scripts/scenario_tool.py new apple_on_block \\
        --object apple=apple@0.24,0.04,30 --object block=real_big_block --step apple:block:left

check builds the real arm workspace scene and reports, for each item, PASS or
FAIL: the file is valid (fields, names, footprints on the worktop and not
overlapping); every object rests where it starts (2 s of physics, no hands);
each step's hand reaches the object and the target (the same simulated test
as test_prop_tasks); and, with --scripted, the scripted demo labels SUCCESS.
--snapshot renders the overview and a top view of the layout.

new writes assets/scenarios/<name>.json from flags, then runs check. Objects:
  --object ID=ASSET@X,Y[,YAW]   an object folder placed on the table (metres, degrees)
  --object ID=PROP              a built-in prop at its usual spot (e.g. real_plate)
  --extra ...                   the same, on the table but not scored
  --step OBJECT:TARGET:HAND[:upright]   TARGET may be "tray"; HAND left or right
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import scenarios

# Wrist goal = object position minus this (palm-down hand: fingers +x, thumb
# toward the midline), with the palm 2 cm above the object's top. As test_prop_tasks.
GRASP_XY = {"left": np.array([.10, -.04]), "right": np.array([.10, .04])}
REACH_TOLERANCE = .01      # m between the wrist goal and where the wrist settles
TRAY_TOP = .038            # tray rim height above the table
REST_SECONDS = 2.
REST_MOVE = .01            # m an object may shift while settling
REST_TILT = 5.             # degrees


def _resolve(name_or_path):
    """Register the scenario's folder; return its name (a file path or a name in assets/scenarios)."""
    path = Path(name_or_path)
    if path.suffix == ".json":
        scenarios.register(path.resolve().parent, quiet=True)
        name = path.stem
    else:
        scenarios.register(quiet=True)
        name = str(name_or_path)
    if name in scenarios.ERRORS:
        raise ValueError(scenarios.ERRORS[name])
    if name not in scenarios.LOADED:
        raise ValueError(f"no scenario {name}; have {sorted(scenarios.LOADED) or 'none'}")
    return name


def _target_top(target):
    from workspace_scene import TABLE_TOP, object_half_height
    return TABLE_TOP + (TRAY_TOP if target == "tray" else 2 * object_half_height(target))


def _target_xy(target):
    from workspace_scene import PROPS, TRAY_CENTER
    return np.asarray(TRAY_CENTER if target == "tray" else PROPS[target]["xy"], dtype=float)


def reach_error(scene, side, goal, updates=250):
    """Distance (m) between a wrist goal and where `side`'s wrist settles."""
    from view_dual_arms import advance_physics
    model, data, controller = scene
    controller.reset(data)
    controller.set_target(side, goal)
    for _ in range(updates):
        controller.command(data)
        advance_physics(model, data, controller)
    return float(np.linalg.norm(data.site_xpos[controller.sites[side]] - goal))


def step_goals(obj, target, side):
    """(label, wrist goal) at the pick-up spot and over the target."""
    from workspace_scene import TABLE_TOP, object_half_height, object_start
    clear = 2 * object_half_height(obj) + .02
    return [("pick up", np.r_[object_start(obj)[:2] - GRASP_XY[side], TABLE_TOP + clear]),
            ("place", np.r_[_target_xy(target) - GRASP_XY[side], _target_top(target) + clear])]


def check_rest(scene, names):
    """{name: (moved m, tilt degrees)} after REST_SECONDS of physics with the arms at home."""
    from view_dual_arms import advance_physics
    model, data, controller = scene
    controller.reset(data)
    start = {n: (data.body(n).xpos.copy(), data.body(n).xmat.reshape(3, 3)[:, 2].copy()) for n in names}
    for _ in range(int(REST_SECONDS / (5 * model.opt.timestep))):
        controller.command(data)
        advance_physics(model, data, controller)
    result = {}
    for name in names:
        position, up = start[name]
        now_up = data.body(name).xmat.reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(np.dot(up, now_up), -1, 1)))
        result[name] = (float(np.linalg.norm(data.body(name).xpos - position)), float(tilt))
    return result


def snapshot(model, path, width=960, height=600):
    """Overview camera beside a top view of the table, as one PNG."""
    import mujoco
    from PIL import Image
    from view_dual_arms import configure_camera
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    images = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for view in ("overview", "top"):
            camera = mujoco.MjvCamera()
            if view == "overview":
                configure_camera(camera, overview=True, workspace=True)
            else:
                # Straight down over the near table, operator's way round
                # (the arms at the bottom, robot left on screen left).
                camera.lookat[:] = [.33, 0, .60]
                camera.distance, camera.azimuth, camera.elevation = .90, 0, -89.9
            renderer.update_scene(data, camera)
            images.append(Image.fromarray(renderer.render()))
    sheet = Image.new("RGB", (2 * width, height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, (index * width, 0))
    sheet.save(path)


def run_check(name, snapshot_path=None, scripted=False, out=print):
    """Print a PASS/FAIL report for a registered scenario; return True if all pass."""
    from view_dual_arms import setup
    from workspace_scene import TASKS, task_props, task_steps
    scenario = scenarios.LOADED[name]
    ok = True

    def report(passed, item, detail=""):
        nonlocal ok
        ok &= passed
        out(f"  {'PASS' if passed else 'FAIL'}  {item}{': ' + detail if detail else ''}")

    out(f"Scenario {name} ({scenario['path'].name})")
    report(True, "valid", "fields, names, on the worktop, no overlapping footprints")
    with contextlib.redirect_stdout(io.StringIO()):
        model, data, controller, _ = setup("human", workspace=True, props=task_props(name))
    scene = (model, data, controller)

    placed = [o["prop"] for o in scenario["objects"].values()]
    shown = {o["prop"]: object_id for object_id, o in scenario["objects"].items()}
    shown.update({fixture: fixture for fixture in scenarios.FIXTURES})
    for prop, (moved, tilt) in check_rest(scene, placed).items():
        report(moved <= REST_MOVE and tilt <= REST_TILT, f"{shown[prop]} rests",
               f"moved {moved * 1000:.0f} mm, tilted {tilt:.0f} degrees in {REST_SECONDS:.0f} s"
               + ("" if moved <= REST_MOVE and tilt <= REST_TILT else
                  " (rolls, tips or spawns touching something; try another spot or yaw, "
                  "or prepare the object with --settle / --condim 6)"))

    for step in task_steps(name):
        spec = TASKS[step]
        side, other = spec["hand"], "right" if spec["hand"] == "left" else "left"
        for label, goal in step_goals(spec["object"], spec["target"], side):
            error = reach_error(scene, side, goal)
            detail = f"{side} hand within {error * 1000:.0f} mm"
            if error > REACH_TOLERANCE:
                other_goal = goal + np.r_[GRASP_XY[side] - GRASP_XY[other], 0]
                other_error = reach_error(scene, other, other_goal)
                detail += (f" (needs {REACH_TOLERANCE * 1000:.0f} mm); the {other} hand "
                           + (f"reaches it ({other_error * 1000:.0f} mm)" if other_error <= REACH_TOLERANCE
                              else f"does not either ({other_error * 1000:.0f} mm); move it closer"))
            report(error <= REACH_TOLERANCE, f"{shown[spec['object']]} -> {shown[spec['target']]}: {label}",
                   detail)

    if snapshot_path:
        try:
            snapshot(model, snapshot_path)
            out(f"  Snapshot: {snapshot_path}")
        except Exception as exc:   # no GL context, etc.: a picture is optional
            out(f"  Snapshot skipped: {exc}")

    if scripted:
        from scripted_prop_tasks import scripted_tasks
        if name not in scripted_tasks():
            report(False, "scripted demo", "the scripted hand is left-handed; this scenario has right-hand steps")
        else:
            from scripted_prop_tasks import run
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                _, labels, _ = run(name, folder)
            outcome = (labels or {}).get("outcome", {})
            detail = ("SUCCESS" if outcome.get("success") else f"FAIL ({outcome.get('failure_reason')})") \
                + (f", confidence {outcome['confidence']:.2f}" if "confidence" in outcome else "")
            report(bool(outcome.get("success")), "scripted demo", detail
                   + ("" if outcome.get("success") else
                      '; tune "scripted" grasp/place_offset in the file (a live operator may still manage)'))
    out("All checks passed." if ok else "Some checks failed.")
    return ok


# -- new ---------------------------------------------------------------------------
def parse_object(text, extra=False):
    """ID=ASSET@X,Y[,YAW] or ID=PROP -> scenario object entry."""
    if "=" not in text:
        raise ValueError(f'"{text}": expected ID=ASSET@X,Y[,YAW] or ID=PROP')
    object_id, rest = text.split("=", 1)
    if "@" not in rest:
        entry = {"id": object_id, "prop": rest}
    else:
        asset, where = rest.split("@", 1)
        try:
            values = [float(v) for v in where.split(",")]
        except ValueError:
            raise ValueError(f'"{text}": position must be numbers, X,Y or X,Y,YAW') from None
        if len(values) not in (2, 3):
            raise ValueError(f'"{text}": position must be X,Y or X,Y,YAW')
        entry = {"id": object_id, "asset": asset, "position_m": values[:2],
                 "yaw_degrees": values[2] if len(values) == 3 else 0.}
    if extra:
        entry["extra"] = True
    return entry


def parse_step(text):
    """OBJECT:TARGET:HAND[:upright] -> scenario step entry."""
    parts = text.split(":")
    if len(parts) not in (3, 4) or (len(parts) == 4 and parts[3] != "upright"):
        raise ValueError(f'"{text}": expected OBJECT:TARGET:HAND or OBJECT:TARGET:HAND:upright')
    return {"object": parts[0], "target": parts[1], "hand": parts[2], "upright": len(parts) == 4}


def write_new(name, objects, extras, steps, description="", directory=None, force=False):
    folder = Path(directory) if directory else scenarios.SCENARIOS_DIR
    path = folder / f"{name}.json"
    if path.exists() and not force:
        raise ValueError(f"{path} exists; pass --force to replace it")
    config = {"schema": scenarios.SCHEMA, "version": scenarios.SCHEMA_VERSION, "name": name,
              "description": description,
              "objects": [parse_object(t) for t in objects] + [parse_object(t, True) for t in extras],
              "steps": [parse_step(t) for t in steps]}
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Every scenario with its objects and steps")
    check = commands.add_parser("check", help="Validate, rest, reach (and scripted) checks")
    check.add_argument("scenario", help="Name in assets/scenarios, or a path to a .json file")
    check.add_argument("--snapshot", help="Write an overview + top view PNG here")
    check.add_argument("--scripted", action="store_true", help="Also run the scripted demo (left hand)")
    new = commands.add_parser("new", help="Write a scenario file from flags, then check it")
    new.add_argument("name")
    new.add_argument("--object", action="append", default=[], dest="objects", metavar="ID=ASSET@X,Y[,YAW]")
    new.add_argument("--extra", action="append", default=[], dest="extras", metavar="ID=ASSET@X,Y[,YAW]")
    new.add_argument("--step", action="append", default=[], dest="steps", metavar="OBJECT:TARGET:HAND")
    new.add_argument("--description", default="")
    new.add_argument("--dir", help="Folder (default: assets/scenarios)")
    new.add_argument("--force", action="store_true", help="Replace an existing file")
    new.add_argument("--snapshot")
    args = parser.parse_args(argv)

    if args.command == "list":
        scenarios.register(quiet=True)
        for name, scenario in scenarios.LOADED.items():
            objects = ", ".join(f"{k}{' (extra)' if o['extra'] else ''}" for k, o in scenario["objects"].items())
            steps = ", ".join(f"{s['id']} -> {s['target']} ({s['hand']})" for s in scenario["steps"])
            print(f"{name}: {scenario['description']}\n  objects: {objects}\n  steps: {steps}")
        for name, error in scenarios.ERRORS.items():
            print(f"{name}: ERROR {error}")
        if not scenarios.LOADED and not scenarios.ERRORS:
            print(f"No scenarios in {scenarios.SCENARIOS_DIR}")
        return 1 if scenarios.ERRORS else 0

    try:
        if args.command == "new":
            if not args.objects or not args.steps:
                parser.error("new needs at least one --object and one --step")
            path = write_new(args.name, args.objects, args.extras, args.steps, args.description,
                             args.dir, args.force)
            print(f"Wrote {path}", flush=True)
            try:
                name = _resolve(path)
            except ValueError as exc:
                print(f"Scenario error: {exc}\nThe file is kept; fix it and run: "
                      f"scenario_tool.py check {path}", file=sys.stderr)
                return 2
            return 0 if run_check(name, args.snapshot) else 1
        name = _resolve(args.scenario)
    except ValueError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2
    return 0 if run_check(name, args.snapshot, args.scripted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
