"""Scenario files (scenarios.py), scenario_tool.py and the object preview."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

import scenarios
import workspace_scene as ws
from test_prop_tasks import label_plate, place_script

GOOD = {
    "schema": "lab3l.scenario", "version": 1, "name": "test_fruit",
    "description": "test",
    "objects": [
        {"id": "apple", "asset": "apple", "position_m": [0.24, 0.04], "yaw_degrees": 0},
        {"id": "banana", "asset": "banana", "position_m": [0.36, 0.30], "yaw_degrees": 20},
        {"id": "plate", "prop": "real_plate"},
    ],
    "steps": [{"object": "apple", "target": "plate", "hand": "left"}],
}


def renderer_available():
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(model, height=16, width=16).close()
        return True
    except Exception:
        return False


class ScenarioFolder(unittest.TestCase):
    """Each test loads scenarios from its own temporary folder, never assets/scenarios."""
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(scenarios.register, None, True)   # back to the repo's files

    def write(self, config, name=None):
        name = name or config["name"]
        path = self.folder / f"{name}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def load_error(self, config, file_name="test_fruit"):
        path = self.write(config, file_name)
        with self.assertRaises(ValueError) as caught:
            scenarios.load(path)
        return str(caught.exception)


class LoadTests(ScenarioFolder):
    def test_single_step_registers_a_task_and_props(self):
        config = copy.deepcopy(GOOD)
        config["objects"][1]["extra"] = True
        self.write(config)
        self.assertEqual(scenarios.register(self.folder, quiet=True), ["test_fruit"])
        task = ws.TASKS["test_fruit"]
        self.assertEqual((task["object"], task["target"], task["hand"], task["upright"]),
                         ("test_fruit_apple", "real_plate", "left", False))
        self.assertEqual(ws.PROPS["test_fruit_banana"]["yaw"], 20)
        self.assertEqual(ws.PROPS["test_fruit_apple"]["xy"], (.24, .04))
        # The extra is on the table but not scored.
        self.assertEqual(ws.task_props("test_fruit"), ["test_fruit_apple", "real_plate", "test_fruit_banana"])
        config["objects"][1].pop("extra")   # unused objects are on the table either way
        self.write(config)
        scenarios.register(self.folder, quiet=True)
        self.assertIn("test_fruit_banana", ws.task_props("test_fruit"))
        self.assertEqual(ws.task_held_objects("test_fruit"), ["test_fruit_apple"])

    def test_several_steps_become_a_step_task(self):
        config = copy.deepcopy(GOOD)
        config["steps"].append({"object": "banana", "target": "plate", "hand": "left"})
        self.write(config)
        scenarios.register(self.folder, quiet=True)
        self.assertEqual(ws.task_steps("test_fruit"), ["test_fruit_apple", "test_fruit_banana"])
        self.assertEqual(ws.TASKS["test_fruit_banana"]["part_of"], "test_fruit")
        from studio.session import tasks
        with mock.patch.object(scenarios, "SCENARIOS_DIR", self.folder):
            offered = tasks()
        self.assertIn("test_fruit", offered)
        self.assertNotIn("test_fruit_apple", offered)   # steps are not offered alone

    def test_reregister_removes_old_entries(self):
        path = self.write(GOOD)
        scenarios.register(self.folder, quiet=True)
        path.unlink()
        scenarios.register(self.folder, quiet=True)
        self.assertNotIn("test_fruit", ws.TASKS)
        self.assertNotIn("test_fruit_apple", ws.PROPS)
        self.assertIn("real_plate", ws.PROPS)   # built-ins untouched

    def test_broken_file_is_skipped_and_reported(self):
        self.write(GOOD)
        bad = copy.deepcopy(GOOD)
        bad["name"] = "broken"
        bad["steps"][0]["hand"] = "both"
        self.write(bad)
        self.assertEqual(scenarios.register(self.folder, quiet=True), ["test_fruit"])
        self.assertIn("hand", scenarios.ERRORS["broken"])
        from view_dual_arms import parse_args
        argv = ["view_dual_arms.py", "--wrist", "human", "--workspace", "--task", "broken"]
        with mock.patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit):
            parse_args()
        self.assertIn("scenario file has an error", err.getvalue())

    def test_errors_name_the_field(self):
        def changed(edit):
            config = copy.deepcopy(GOOD)
            edit(config)
            return config
        cases = {
            "schema": changed(lambda c: c.update(schema="other")),
            "match the file name": changed(lambda c: c.update(name="other_name")),
            "already a task": changed(lambda c: c.update(name="apple_to_plate")),
            'objects[0].asset "nope"': changed(lambda c: c["objects"][0].update(asset="nope")),
            "not a built-in prop": changed(lambda c: c["objects"][2].update(prop="tray")),
            "used twice": changed(lambda c: c["objects"].append(dict(c["objects"][0]))),
            "reserved": changed(lambda c: c["objects"][0].update(id="cup")),
            "position_m must be [x, y]": changed(lambda c: c["objects"][0].update(position_m=[.2, .1, .6])),
            "not allowed with prop": changed(lambda c: c["objects"][2].update(position_m=[.2, .1])),
            "not fully on the worktop": changed(lambda c: c["objects"][0].update(position_m=[.10, .04])),
            "footprints overlap": changed(lambda c: c["objects"][0].update(position_m=[.24, .18])),
            "tray and apple": changed(lambda c: c["objects"][0].update(position_m=[.30, -.10])),
            'steps[0].target "bowl"': changed(lambda c: c["steps"][0].update(target="bowl")),
            "marked extra": changed(lambda c: c["objects"][2].update(extra=True)),
            "already moved": changed(lambda c: c["steps"].append(dict(c["steps"][0]))),
            "hand must be": changed(lambda c: c["steps"][0].update(hand="both")),
            "same": changed(lambda c: c["steps"][0].update(target="apple")),
            "finite number": changed(lambda c: c["objects"][0].update(yaw_degrees="ten")),
            "unknown field": changed(lambda c: c.update(colour="red")),
            "scripted.grasp.apple.offset": changed(lambda c: c.update(scripted={"grasp": {"apple": {"offset": [1]}}})),
        }
        for message, config in cases.items():
            with self.subTest(message):
                # The clash case needs a matching file name to get past that check.
                file_name = config["name"] if message == "already a task" else "test_fruit"
                self.assertIn(message, self.load_error(config, file_name))

    def test_cup_and_tray_are_usable(self):
        config = copy.deepcopy(GOOD)
        config["objects"] = [{"id": "apple", "asset": "apple", "position_m": [.36, -.36]}]
        config["steps"] = [{"object": "apple", "target": "tray", "hand": "right"}]
        self.write(config)
        scenarios.register(self.folder, quiet=True)
        self.assertEqual(ws.TASKS["test_fruit"]["target"], "tray")

    def test_summary_and_replay_warning(self):
        path = self.write(GOOD)
        scenarios.register(self.folder, quiet=True)
        summary = scenarios.summary("test_fruit")
        self.assertEqual(summary["name"], "test_fruit")
        self.assertIsNone(scenarios.summary("apple_to_plate"))
        from replay_demo import asset_warnings
        metadata = {"scene": {"scenario": summary}}
        self.assertEqual(asset_warnings(metadata), [])
        config = copy.deepcopy(GOOD)
        config["objects"][0]["position_m"] = [.25, .04]
        path.write_text(json.dumps(config), encoding="utf-8")
        scenarios.register(self.folder, quiet=True)
        self.assertTrue(any("changed since" in w for w in asset_warnings(metadata)))


class SceneTests(ScenarioFolder):
    def test_yaw_turns_the_body_and_labels_still_score(self):
        config = copy.deepcopy(GOOD)
        config["objects"][1]["yaw_degrees"] = 90
        self.write(config)
        scenarios.register(self.folder, quiet=True)
        from view_dual_arms import setup
        with contextlib.redirect_stdout(io.StringIO()):
            model, data, _, _ = setup("human", workspace=True, props=ws.task_props("test_fruit"))
        quat = model.body("test_fruit_banana").quat
        np.testing.assert_allclose(quat, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-9)
        # A synthetic pick-and-place of the scenario's apple onto the plate scores as success.
        script = place_script("test_fruit_apple", "real_plate")
        labels = label_plate(script, "test_fruit_apple", "test_fruit")
        self.assertTrue(labels["outcome"]["success"], labels["outcome"])

    def test_check_passes_good_and_fails_unreachable(self):
        import scenario_tool
        self.write(GOOD)
        lines = []
        scenarios.register(self.folder, quiet=True)
        self.assertTrue(scenario_tool.run_check("test_fruit", out=lines.append), "\n".join(lines))
        far = copy.deepcopy(GOOD)
        far["name"] = "test_far"
        far["objects"][0]["position_m"] = [.95, .40]
        self.write(far)
        scenarios.register(self.folder, quiet=True)
        lines = []
        self.assertFalse(scenario_tool.run_check("test_far", out=lines.append))
        self.assertTrue(any("FAIL  apple -> plate: pick up" in line for line in lines), lines)

    def test_new_writes_a_file_from_flags(self):
        import scenario_tool
        path = scenario_tool.write_new("test_new", ["apple=apple@0.24,0.04,15", "plate=real_plate"],
                                       ["banana=banana@0.36,0.30"], ["apple:plate:left"], "d", self.folder)
        loaded = scenarios.load(path)
        self.assertEqual(loaded["objects"]["apple"]["spec"]["yaw"], 15)
        self.assertTrue(loaded["objects"]["banana"]["extra"])
        self.assertEqual(loaded["objects"]["plate"]["prop"], "real_plate")
        with self.assertRaises(ValueError):
            scenario_tool.write_new("test_new", ["apple=apple@0.24,0.04"], [], ["apple:tray:right"],
                                    directory=self.folder)   # exists
        with self.assertRaises(ValueError):
            scenario_tool.parse_step("apple>plate")


class CommittedScenarioTests(unittest.TestCase):
    def test_repo_scenarios_load(self):
        scenarios.register(quiet=True)
        self.assertEqual(scenarios.ERRORS, {})
        self.assertIn("fruit_to_plate", scenarios.LOADED)


@unittest.skipUnless(renderer_available(), "mujoco.Renderer unavailable (no GL context)")
class PreviewTests(unittest.TestCase):
    def test_preview_only_writes_just_the_png(self):
        from object_preview import render_source
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "preview.png"
            size = render_source(Path(ws.PROPS_DIR) / "objects/banana/source/object_0.glb", out,
                                 up_axis="y", width_m=.19, rotate_z=90)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["preview.png"])
            np.testing.assert_allclose(size[1], .19, atol=1e-6)
            from PIL import Image
            self.assertEqual(Image.open(out).size[0], 3 * 480)

    def test_folder_preview_adds_the_collision_panel(self):
        from object_preview import render_folder
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            out = render_folder("banana", Path(folder) / "banana.png")
            self.assertEqual(Image.open(out).size[0], 4 * 480)

    def test_cli_preview_only_needs_no_destination(self):
        from prepare_object import main
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as out:
            code = main([str(Path(ws.PROPS_DIR) / "objects/apple/source/object_0-2.glb"),
                         "--preview-only", str(Path(folder) / "a.png"), "--up-axis", "y", "--height", ".06"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing else was written", out.getvalue())


if __name__ == "__main__":
    unittest.main()
