"""Headless reachability probe for the human-arm cup-to-tray workspace.

Sweeps wrist targets over each arm's clip box (HumanArmController.set_target)
at three wrist orientations, settles physics at every target, and measures
how close each hand gets to the cup and tray. It reports the best palm-inward
"wrap" pose for the cup and whether a cup held in that pose could be lowered
into the tray. The model is built from the current workspace_scene constants,
so re-run this after moving the props.

The hand stays open and the cup and tray are made non-colliding (the cup is
held in place), so distances are signed geometry gaps, not grasp results.
No viewer, no headset; simulation only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from arm_control import SIDES, rotation_matrix
from episode_recorder import CONTACT_GROUPS, ContactTables


# set_target's clip box (offset from home) and reach cap, per side.
CLIP_LOW = {"left": np.array([-.16, -.30, -.16]), "right": np.array([-.16, -.14, -.16])}
CLIP_HIGH = {"left": np.array([.16, .14, .18]), "right": np.array([.16, .30, .18])}
REACH_CAP = .53
ORIENTATIONS = ("palm_down", "palm_inward", "palm_inward_pitch30")
WRAP_DISTANCE = .01   # m; thumb and a finger group this close to the cup wall
WRAP_BAND = .02       # m; ... at the cup's mid-height +/- this
MAX_PENETRATION = .005  # m; deeper hand-cup overlap is not a usable wrap
MIN_CLEARANCE = .03   # m from every clip/reach limit
SETTLED_ERROR = .01   # m; achieved wrist within this of the clipped target
REFINE_COUNT = 12     # coarse points per side/orientation refined at half the step
FINGERS = [CONTACT_GROUPS.index(name) for name in ("ff", "mf", "rf", "lf")]
THUMB = CONTACT_GROUPS.index("thumb")
PALM = CONTACT_GROUPS.index("palm")


def orientation(controller, side, name):
    """Wrist rotation target: home palm-down, or palms turned toward the midline.

    Palm-inward is a +/-90 degree turn about the forearm (world +X at home),
    the same axis U/P uses; the pitch variant then tips the fingers 30 degrees
    down about world Y.
    """
    home = controller.home_rotations[side]
    if name == "palm_down":
        return home.copy()
    turn = rotation_matrix([np.pi / 2 if side == "right" else -np.pi / 2, 0, 0]) @ home
    if name == "palm_inward":
        return turn
    if name == "palm_inward_pitch30":
        return rotation_matrix([0, np.deg2rad(30), 0]) @ turn
    raise ValueError(f"Unknown orientation {name!r}")


def limit_clearance(controller, side, target):
    """Distance (m) from target to the nearest clip-box face or the reach cap.

    Negative when set_target clips the target (including the elbow-envelope
    shortening for far inward reaches).
    """
    target = np.asarray(target, dtype=float)
    controller.set_target(side, target)
    if controller.reach_errors[side] > 1e-9:
        return -float(controller.reach_errors[side])
    offset = target - controller.home[side]
    shoulder = controller.model.body_pos[controller.shoulder_b[side]]
    return float(min(np.min(offset - CLIP_LOW[side]), np.min(CLIP_HIGH[side] - offset),
                     REACH_CAP - np.linalg.norm(target - shoulder)))


class Probe:
    """One workspace model with the cup frozen and cup/tray contacts disabled."""

    def __init__(self):
        from view_dual_arms import setup
        self.model, self.data, self.controller, self.hands = setup("human", workspace=True)
        model = self.model
        tables = ContactTables(model)
        colliding = (model.geom_contype | model.geom_conaffinity) > 0
        self.hand = {side: np.flatnonzero((tables.hand_group[side] >= 0) & colliding) for side in SIDES}
        self.group = {side: tables.hand_group[side][self.hand[side]] for side in SIDES}
        self.cup = np.flatnonzero(tables.cup)
        self.cup_walls = np.array([model.geom(f"cup_wall_{i}").id for i in range(24)])
        self.tray = np.flatnonzero(tables.tray)
        self.table = np.array([model.geom("worktop").id])
        for geom in np.r_[self.cup, self.tray]:
            model.geom_contype[geom] = model.geom_conaffinity[geom] = 0
        cup_body = model.body("cup").id
        model.body_gravcomp[cup_body] = 1
        joint = model.joint("cup_free")
        self.cup_q = int(joint.qposadr[0])
        self.cup_v = int(joint.dofadr[0])
        self.fromto = np.zeros(6)

    def reset(self):
        self.controller.reset(self.data)

    def settle(self, side, target, rotation, seconds, minimum=.3):
        """Step physics until the wrist stops moving, at most `seconds` of sim time."""
        from view_dual_arms import advance_physics
        model, data, controller = self.model, self.data, self.controller
        controller.set_target(side, target)
        controller.set_rotation(side, rotation)
        site = controller.sites[side]
        start = data.time
        previous = data.site_xpos[site].copy()
        while data.time - start < seconds:
            controller.command(data, active=[side])
            advance_physics(model, data, controller)
            data.qvel[self.cup_v:self.cup_v + 6] = 0
            moved = np.linalg.norm(data.site_xpos[site] - previous)
            previous = data.site_xpos[site].copy()
            if data.time - start >= minimum and moved < 1e-5:
                break
        return data.site_xpos[site].copy()

    def distance(self, hand_geoms, others, distmax=.08):
        """Min signed distance and the nearest point on `others`."""
        best, point = distmax, None
        for g in hand_geoms:
            for o in others:
                gap = mujoco.mj_geomDistance(self.model, self.data, int(g), int(o), distmax, self.fromto)
                if gap < best:
                    best, point = gap, self.fromto[3:].copy()
        return float(best), point

    def measure(self, side):
        """Per-group hand-cup distances (and nearest cup-point heights), tray, table."""
        cup_z = float(self.data.qpos[self.cup_q + 2])
        groups = []
        for index in range(len(CONTACT_GROUPS) - 1):
            geoms = self.hand[side][self.group[side] == index]
            gap, point = self.distance(geoms, self.cup_walls)
            groups.append((gap, None if point is None else float(point[2] - cup_z)))
        whole, _ = self.distance(self.hand[side], self.cup)
        tray, _ = self.distance(self.hand[side], self.tray)
        table, _ = self.distance(self.hand[side], self.table)
        return dict(cup=whole, cup_groups=[g for g, _ in groups], cup_heights=[h for _, h in groups],
                    tray=tray, table=table)


def grid_axis(low, high, step):
    return np.round(np.arange(low, high + step / 2, step), 6)


def probe_point(probe, side, name, offset, rotation, settle):
    """Clip, settle and measure one wrist target (offset from home)."""
    controller = probe.controller
    clearance = limit_clearance(controller, side, controller.home[side] + offset)
    target = controller.targets[side].copy()
    record = dict(side=side, orientation=name, offset=[round(float(v), 6) for v in offset],
                  target=target.tolist(), reach_error=float(controller.reach_errors[side]),
                  clearance=clearance)
    if clearance < -1e-9:
        # Clipped targets are not settled: the arm would stop at the limit.
        return record | dict(settled=False)
    achieved = probe.settle(side, target, rotation, settle)
    record.update(settled=True, achieved=achieved.tolist(),
                  error=float(np.linalg.norm(achieved - target)), **probe.measure(side))
    return record


def start_probe(side, name):
    probe = Probe()
    probe.reset()
    rotation = orientation(probe.controller, side, name)
    # Turn the wrist at home first so every point starts from the same posture.
    probe.settle(side, probe.controller.home[side].copy(), rotation, 1.5, minimum=1.)
    return probe, rotation


def probe_slice(job):
    """Worker: one (side, orientation, dx) slice, serpentine-ordered for warm starts."""
    side, name, dx, step, settle = job
    probe, rotation = start_probe(side, name)
    records = []
    dys = grid_axis(CLIP_LOW[side][1], CLIP_HIGH[side][1], step)
    for k, dz in enumerate(grid_axis(CLIP_LOW[side][2], CLIP_HIGH[side][2], step)):
        for dy in (dys if k % 2 == 0 else dys[::-1]):
            records.append(probe_point(probe, side, name, np.array([dx, dy, dz]), rotation, settle))
    return records


def refine_point(job):
    """Worker: a 3x3x3 grid at half the step around one coarse grid point."""
    side, name, centre, step, settle = job
    probe, rotation = start_probe(side, name)
    probe.settle(side, probe.controller.home[side] + centre, rotation, settle)
    records = []
    for delta in np.stack(np.meshgrid(*[[-step / 2, 0, step / 2]] * 3, indexing="ij"), -1).reshape(-1, 3):
        if not delta.any():
            continue
        records.append(probe_point(probe, side, name, np.asarray(centre) + delta, rotation, settle)
                       | dict(refined=True))
    return records


def refine_candidates(records, count=REFINE_COUNT):
    """Coarse palm-inward points that nearly wrap the cup, to refine.

    Ranked by how many hand groups are within WRAP_DISTANCE of the cup, then
    by overlap; seeds may overlap the cup a little, since half a step away
    they may not.
    """
    jobs = []
    for side in SIDES:
        for name in ORIENTATIONS[1:]:
            rows = [r for r in records if r["side"] == side and r["orientation"] == name
                    and r.get("settled") and -3 * MAX_PENETRATION <= r["cup"] < WRAP_DISTANCE]
            near = lambda r: sum(gap <= WRAP_DISTANCE for gap in r["cup_groups"])
            rows.sort(key=lambda r: (-near(r), abs(r["cup"])))
            jobs += [(side, name, r["offset"]) for r in rows[:count]]
    return jobs


def wrap_kind(record):
    """"strict" (spec: thumb and a finger group at mid-height), "rim" or None.

    With the Shadow hand turned palm-inward the open thumb lies about 8 cm
    above the finger plane, so its nearest cup point is usually the top rim.
    "rim" accepts that: a finger group at mid-height, and palm and thumb within
    WRAP_DISTANCE of the cup anywhere (a flat palm's nearest point along a
    line contact has no meaningful height; the thumb is in practice on the rim).
    """
    if not record.get("settled") or record["orientation"] == "palm_down":
        return None
    if record["error"] > SETTLED_ERROR or record["cup"] < -MAX_PENETRATION:
        return None
    gaps, heights = record["cup_groups"], record["cup_heights"]
    def near(group, band=True):
        return gaps[group] <= WRAP_DISTANCE and heights[group] is not None and (
            not band or abs(heights[group]) <= WRAP_BAND)
    if not any(near(group) for group in FINGERS):
        return None
    if near(THUMB):
        return "strict"
    if near(THUMB, band=False) and near(PALM, band=False):
        return "rim"
    return None


def is_wrap(record):
    return wrap_kind(record) == "strict"


def placement(controller, side, wrap, cup_start, tray_center, tray_rest, inner, step=.01):
    """Best cup position inside the tray if the wrap pose's wrist-to-cup offset is kept."""
    offset = np.asarray(cup_start) - np.asarray(wrap["achieved"])
    best = None
    for x in grid_axis(-inner[0], inner[0], step):
        for y in grid_axis(-inner[1], inner[1], step):
            cup = np.array([tray_center[0] + x, tray_center[1] + y, tray_rest])
            clearance = limit_clearance(controller, side, cup - offset)
            # Prefer limit clearance, then cups near the tray centre.
            score = (round(clearance, 4), -np.hypot(x, y))
            if best is None or score > best[0]:
                best = (score, dict(cup=cup.tolist(), wrist=(cup - offset).tolist(),
                                    clearance=clearance))
    centre = np.array([*tray_center, tray_rest])
    return best[1], limit_clearance(controller, side, centre - offset)


def summarize(records, sides):
    from workspace_scene import (CUP_HALF_HEIGHT, CUP_RADIUS, CUP_START, TABLE_TOP, TRAY_CENTER,
                                 TRAY_HALF_EXTENTS, TRAY_RIM_THICKNESS)
    probe = Probe()
    controller = probe.controller
    tray_rest = TABLE_TOP + .012 + CUP_HALF_HEIGHT
    # Cup centre inside the inner rim, with the cup body clear of the rim.
    inner = np.asarray(TRAY_HALF_EXTENTS) - TRAY_RIM_THICKNESS - CUP_RADIUS
    report = dict(scene=dict(table_top=TABLE_TOP, cup_start=CUP_START.tolist(),
                             tray_center=TRAY_CENTER.tolist()), sides={})
    lines = [f"Scene: table_top {TABLE_TOP:.3f}, cup_start {np.round(CUP_START, 3).tolist()}, "
             f"tray_center {np.round(TRAY_CENTER, 3).tolist()}"]
    for side in sides:
        mine = [r for r in records if r["side"] == side]
        settled = [r for r in mine if r.get("settled")]
        home = controller.home[side]
        refined = sum(1 for r in mine if r.get("refined"))
        lines.append(f"\n{side}: home {np.round(home, 3).tolist()}; {len(mine) - refined} grid targets "
                     f"+ {refined} refined near the cup, {len(settled)} inside the limits and settled")
        summary = dict(home=home.tolist(), targets=len(mine), settled=len(settled))
        if settled:
            low = min(settled, key=lambda r: r["achieved"][2])
            summary["lowest_wrist_z"] = low["achieved"][2]
            lines.append(f"  lowest settled wrist z {low['achieved'][2]:.3f} "
                         f"(offset {np.round(low['offset'], 2).tolist()}, {low['orientation']})")
            for name in ORIENTATIONS:
                rows = [r for r in settled if r["orientation"] == name]
                if rows:
                    closest = min(rows, key=lambda r: r["cup"])
                    lines.append(f"  {name:20s} closest hand-cup gap {closest['cup'] * 100:6.1f} cm "
                                 f"at wrist {np.round(closest['achieved'], 3).tolist()}, "
                                 f"limit clearance {closest['clearance'] * 100:.1f} cm")
        strict = [r for r in settled if wrap_kind(r) == "strict"]
        rim = [r for r in settled if wrap_kind(r) == "rim"]
        summary.update(wraps=len(strict), rim_wraps=len(rim))
        lines.append(f"  wrap poses: {len(strict)} strict (thumb + finger within "
                     f"{WRAP_DISTANCE * 100:.0f} cm of the wall at mid-height +/- {WRAP_BAND * 100:.0f} cm), "
                     f"{len(rim)} with the thumb on the rim instead")
        wraps = strict or rim
        if not wraps:
            summary["best_wrap"] = None
            report["sides"][side] = summary
            continue
        best = max(wraps, key=lambda r: (r["clearance"], -r["cup"]))
        summary["best_wrap_kind"] = wrap_kind(best)
        place, centre = placement(controller, side, best, CUP_START, TRAY_CENTER, tray_rest, inner)
        summary.update(best_wrap=best, placement=place, tray_centre_clearance=centre,
                       wrap_ok=best["clearance"] >= MIN_CLEARANCE,
                       place_ok=place["clearance"] >= MIN_CLEARANCE)
        groups = ", ".join(f"{name} {gap * 100:.1f}" for name, gap in
                           zip(CONTACT_GROUPS, best["cup_groups"]))
        lines += [
            f"  best {wrap_kind(best)} wrap: {best['orientation']}, wrist "
            f"{np.round(best['achieved'], 3).tolist()} (offset {np.round(best['offset'], 2).tolist()}), "
            f"limit clearance {best['clearance'] * 100:.1f} cm "
            f"[{'OK' if summary['wrap_ok'] else 'below'} {MIN_CLEARANCE * 100:.0f} cm]",
            f"    hand-cup gaps (cm): {groups}; table gap {best['table'] * 100:.1f} cm",
            f"  holding that pose, best cup spot in the tray {np.round(place['cup'], 3).tolist()}: "
            f"limit clearance {place['clearance'] * 100:.1f} cm "
            f"[{'OK' if summary['place_ok'] else 'below'} {MIN_CLEARANCE * 100:.0f} cm]; "
            f"at the tray centre {centre * 100:.1f} cm",
        ]
        report["sides"][side] = summary
    return report, "\n".join(lines)


def run(sides=SIDES, step=.02, settle=3., workers=None, orientations=ORIENTATIONS):
    """Coarse grid, then a half-step refinement around the closest palm-inward points.

    A wrap needs gaps within 1 cm, finer than the 2 cm grid, so the
    refinement is what usually finds it.
    """
    jobs = []
    for side in sides:
        for name in orientations:
            for dx in grid_axis(CLIP_LOW[side][0], CLIP_HIGH[side][0], step):
                jobs.append((side, name, float(dx), step, settle))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    def gather(function, items):
        if workers == 1:
            return [r for chunk in map(function, items) for r in chunk]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return [r for chunk in pool.map(function, items) for r in chunk]
    records = gather(probe_slice, jobs)
    fine = [(side, name, offset, step, settle) for side, name, offset in refine_candidates(records)]
    return records + gather(refine_point, fine)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=(*SIDES, "both"), default="both")
    parser.add_argument("--step", type=float, default=.02, help="Grid step in metres (default 0.02)")
    parser.add_argument("--settle", type=float, default=3.,
                        help="Max sim seconds per target; stops early once the wrist is still (default 3)")
    parser.add_argument("--workers", type=int, default=0, help="Processes (default: CPU count - 1)")
    parser.add_argument("--json", help="Write every grid point and the summary to this file")
    args = parser.parse_args(argv)
    if not np.isfinite(args.step) or not .005 <= args.step <= .1:
        parser.error("--step must be between 0.005 and 0.1")
    if not np.isfinite(args.settle) or args.settle <= 0:
        parser.error("--settle must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    sides = SIDES if args.side == "both" else (args.side,)
    started = time.monotonic()
    records = run(sides, args.step, args.settle, args.workers or None)
    report, text = summarize(records, sides)
    print(text)
    print(f"\n{len(records)} grid points in {time.monotonic() - started:.0f} s.")
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(report, grid=records), indent=1))
        print(f"Wrote {path}")
    return report


if __name__ == "__main__":
    main()
