"""Committed real 3D objects (assets/objects): valid, credited, and fast enough."""
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import object_assets
from workspace_scene import PROP_GROUPS, PROPS, expand_props

FRAME_BUDGET_MS = 16      # view_dual_arms --headset-budget-ms default
LOAD_BUDGET_S = 2.0


class CommittedObjectTests(unittest.TestCase):
    def test_every_object_folder_is_valid_and_credited(self):
        folders = sorted(p for p in object_assets.OBJECTS_DIR.iterdir() if p.is_dir())
        self.assertTrue(folders)
        for folder in folders:
            with self.subTest(folder=folder.name):
                config = object_assets.load(folder)
                self.assertEqual(config["id"], folder.name)
                self.assertTrue(set(config["values"].values()) <= set(object_assets.VALUE_ORIGINS))
                source = config["source"]
                if source["kind"] in ("scanned_dataset", "external_mesh"):
                    # Third-party meshes are redistributed only with their license.
                    self.assertTrue(source.get("license"), folder.name)
                    self.assertTrue(source.get("attribution"), folder.name)
                self.assertIn(config["footprint"]["shape"], ("round", "box"))

    def test_real_props_point_at_committed_objects(self):
        for name in PROP_GROUPS["real"] + PROP_GROUPS["sam"]:
            spec = PROPS[name]
            self.assertTrue((object_assets.OBJECTS_DIR / spec["asset"] / "object.json").is_file(), name)
            if "replaces" in spec:   # a copy of a primitive prop sits on its spot
                self.assertIn(spec["replaces"], PROPS)
                self.assertEqual(spec["xy"], PROPS[spec["replaces"]]["xy"])

    def test_prop_groups_and_clashes(self):
        self.assertEqual(expand_props(["all"]), PROP_GROUPS["all"])
        self.assertEqual(expand_props(["real", "real_apple"]), PROP_GROUPS["real"])
        with self.assertRaisesRegex(ValueError, "share a spot"):
            expand_props(["apple", "real_apple"])
        with self.assertRaisesRegex(ValueError, "share a spot"):
            expand_props(["sam_apple", "real_apple"])   # same spot, neither replaces the other
        with self.assertRaisesRegex(ValueError, "share a spot"):
            expand_props(["banana", "block"])           # declared overlap
        with self.assertRaisesRegex(ValueError, "Unknown"):
            expand_props(["mug"])

    def test_digest_changes_with_any_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "ycb_block_5cm"
            shutil.copytree(object_assets.OBJECTS_DIR / "ycb_block_5cm", copy)
            before = object_assets.digest(copy)
            self.assertEqual(before, object_assets.digest(object_assets.OBJECTS_DIR / "ycb_block_5cm"))
            with open(copy / "visual_0.obj", "a") as file:
                file.write("# edited\n")
            self.assertNotEqual(before, object_assets.digest(copy))

    def test_load_time_and_sim_loop_within_frame_budget(self):
        """Every real model on the table while a scripted hand handles them.

        A control update (five 2 ms physics steps) must fit the headset frame
        budget. The 95th percentile is checked: single outliers are OS
        scheduling noise, not physics cost (rerunning the same deterministic
        steps does not reproduce them).
        """
        import scripted_prop_tasks as spt
        import view_dual_arms
        from scripted_cup_to_tray import Script
        start = time.perf_counter()
        model, data, controller, hands = view_dual_arms.setup("human", workspace=True,
                                                              props=PROP_GROUPS["real"])
        load = time.perf_counter() - start
        self.assertLess(load, LOAD_BUDGET_S)
        script = Script(model, data, controller, hands, None, None, side=spt.SIDE)
        original, times = view_dual_arms.advance_physics, []

        def timed(*args, **kwargs):
            tick = time.perf_counter()
            original(*args, **kwargs)
            times.append(time.perf_counter() - tick)
        view_dual_arms.advance_physics = timed
        try:
            spt.place_object(script, "real_apple", "real_plate", (0, -.03))
        finally:
            view_dual_arms.advance_physics = original
        loop_ms = np.array(times) * 1000
        self.assertGreater(len(loop_ms), 100)
        self.assertLess(np.percentile(loop_ms, 95), FRAME_BUDGET_MS)
        self.assertLess(np.median(loop_ms), FRAME_BUDGET_MS / 4)


if __name__ == "__main__":
    unittest.main()
