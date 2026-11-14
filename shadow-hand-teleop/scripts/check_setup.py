"""Verify Shadow Hand setup; use --include-arms for Panda and --offline for no downloads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys

import mujoco
import mujoco_menagerie as menagerie
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MENAGERIE_CACHE_DIR",
    str(PROJECT_ROOT / ".cache" / "mujoco_menagerie"),
)


def check_model(name, cache, *, offline=False):
    if offline and cache.root is None and not cache.is_cached(menagerie.get(name)):
        raise RuntimeError("model assets are not cached; offline mode does not download them")
    # MENAGERIE_ROOT is resolved locally by Menagerie and never downloads assets.
    model = menagerie.load(name, cache=cache)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    for _ in range(250):
        mujoco.mj_step(model, data)

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("simulation produced invalid joint positions or velocities")

    print(f"{name}: {model.nu} actuators, {model.nq} position coordinates")
    print(f"Scene: {model.nbody} bodies, {model.ngeom} collision/visual geometries")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-arms", action="store_true",
                        help="Also validate the Panda model used by the arm variants")
    parser.add_argument("--offline", action="store_true",
                        help="Use local model assets only; never attempt a download")
    args = parser.parse_args(argv)
    names = ["shadow_hand"]
    if args.include_arms:
        names.append("franka_emika_panda")
    cache = menagerie.Cache()
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Model assets: {cache.root if cache.root is not None else cache.dir}")
    failed = []
    for name in names:
        try:
            check_model(name, cache, offline=args.offline)
        except (menagerie.MenagerieError, OSError, ValueError, RuntimeError) as exc:
            failed.append(name)
            print(f"Setup check failed for {name}: {exc}", file=sys.stderr)
    if failed:
        if cache.root is not None:
            print("Check MENAGERIE_ROOT: it must contain complete model directories named "
                  + ", ".join(failed) + ".", file=sys.stderr)
        else:
            code = f"import mujoco_menagerie as m; m.prefetch({failed!r})"
            command = (f"MENAGERIE_CACHE_DIR={shlex.quote(str(cache.dir))} "
                       f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}")
            print("Download the required assets while online, then rerun this check:\n  "
                  + command, file=sys.stderr)
            print("If assets are already present, check their files and the model error above.",
                  file=sys.stderr)
        return 1
    print("Setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

