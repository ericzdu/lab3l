"""--scene-config: object folders, version-2 scenes, legacy bundles, recording/replay.

No SAM inference or headset-validation claims.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import mujoco
import numpy as np
import trimesh

from photo_scene import add_photo_scene, load_bundle, object_names, scene_digest, scene_summary
from prepare_object import prepare


PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT / "assets/photo_scene_test/scene.json"


class PhotoSceneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "box.glb"
        trimesh.creation.box(extents=[2, 4, 6]).export(source)
        self.folder = prepare(source, self.root / "scene/objects/thing", object_id="thing",
                              up_axis="z", height_m=.12, mass_kg=.2, collision="convex",
                              position=(.55, .28, .60)).parent

    def scene(self, objects):
        path = self.root / "scene/scene.json"
        path.write_text(json.dumps({"version": 2, "objects": objects}), encoding="utf-8")
        return path

    def build(self, path):
        spec = mujoco.MjSpec()
        add_photo_scene(spec, path)
        return spec.compile()

    def test_object_folder_uses_its_own_placement(self):
        model = self.build(self.folder)
        self.assertEqual(object_names(self.folder), ["photo_thing"])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Placement is the bottom centre; the body origin is the box centre.
        np.testing.assert_allclose(data.body("photo_thing").xpos, [.55, .28, .66], atol=1e-9)
        self.assertAlmostEqual(model.body("photo_thing").mass[0], .2)

    def test_version_2_scene_with_relative_and_id_assets(self):
        path = self.scene([
            {"id": "a", "asset": "objects/thing", "position_m": [.5, .2, .6], "yaw_degrees": 90},
            {"id": "carton", "asset": "test_coffee_carton", "position_m": [.7, .2, .6], "fixed": True}])
        model = self.build(path)
        self.assertEqual(object_names(path), ["photo_a", "photo_carton"])
        self.assertEqual(model.njnt, 1)                       # the fixed carton has no joint
        self.assertEqual(model.body("photo_a").quat.round(6).tolist(),
                         np.round([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], 6).tolist())

    def test_asset_cannot_escape_the_scene(self):
        shutil.copytree(self.folder, self.root / "outside")
        path = self.scene([{"id": "a", "asset": "../outside", "position_m": [.5, .2, .6]}])
        with self.assertRaisesRegex(ValueError, "outside scene bundle"):
            self.build(path)

    def test_object_files_cannot_escape_their_folder(self):
        config = json.loads((self.folder / "object.json").read_text())
        config["visual"][0]["mesh"] = "../../box.glb"
        (self.folder / "object.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "outside object folder"):
            self.build(self.folder)

    def test_duplicate_ids_rejected(self):
        entry = {"id": "a", "asset": "objects/thing", "position_m": [.5, .2, .6]}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build(self.scene([entry, dict(entry)]))

    def test_digest_follows_the_object_files(self):
        path = self.scene([{"id": "a", "asset": "objects/thing", "position_m": [.5, .2, .6]}])
        before = scene_digest(path)
        self.assertEqual(before, scene_digest(path))
        with open(self.folder / "visual_0.obj", "a") as file:
            file.write("# edited\n")
        self.assertNotEqual(before, scene_digest(path))

    def test_legacy_version_1_bundle_still_loads(self):
        """The original prototype's bundles (box/convex/boxes collision) keep working."""
        legacy = self.root / "legacy"
        legacy.mkdir()
        trimesh.creation.box(extents=[.04, .08, .12]).export(legacy / "visual_0.obj")
        config = {"version": 1, "objects": [{
            "id": "old", "parts": [{"mesh": "visual_0.obj"}], "position_m": [.5, .2, .6],
            "mass_kg": .15, "collision": {"mode": "boxes", "boxes": [
                {"position_m": [-.018, 0, .06], "half_size_m": [.002, .04, .06]},
                {"position_m": [.018, 0, .06], "half_size_m": [.002, .04, .06]}]}}]}
        path = legacy / "scene.json"
        path.write_text(json.dumps(config))
        self.assertEqual(len(load_bundle(path)), 1)
        model = self.build(path)
        self.assertAlmostEqual(model.body("photo_old").mass[0], .15)
        self.assertEqual(model.geom("photo_old_collision_1").type[0], mujoco.mjtGeom.mjGEOM_BOX)
        self.assertEqual(object_names(path), ["photo_old"])
        self.assertEqual(len(scene_digest(path)), 64)

    def test_original_arms_workspace_and_imported_prop(self):
        from view_dual_arms import setup, advance_physics
        model, data, controller, _ = setup("human", workspace=True, scene_config=FIXTURE)
        body = model.body("photo_test_coffee_carton").id
        address = int(model.joint("photo_test_coffee_carton_free").qposadr[0])
        initial = data.qpos[address:address + 7].copy()
        self.assertAlmostEqual(model.body_mass[body], .15)
        self.assertEqual(model.body_gravcomp[body], 0)
        data.qpos[address + 2] += .1
        mujoco.mj_forward(model, data)
        for _ in range(80):
            controller.command(data)
            advance_physics(model, data, controller)
        self.assertLess(data.qpos[address + 2], initial[2] + .002)
        self.assertGreater(data.qpos[address + 2], initial[2] - .003)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertLess(np.max(np.abs(data.qvel)), .1)
        controller.reset(data)
        np.testing.assert_allclose(data.qpos[address:address + 7], initial)
        # Existing cup/tray and all original actuators remain present.
        self.assertGreaterEqual(model.body("cup").id, 0)
        self.assertGreaterEqual(model.body("tray").id, 0)
        self.assertEqual(model.nu, 50)

    def test_imported_scene_is_recorded_and_replayed(self):
        from episode_recorder import EpisodeRecorder
        from replay_demo import asset_warnings, episode_model
        from test_episode_recorder import run
        from view_dual_arms import setup
        names = object_names(FIXTURE)
        model, data, controller, hands = setup("human", workspace=True, scene_config=FIXTURE)
        recorder = EpisodeRecorder(model, controller, hands, record_dir=self.root / "episodes",
                                   source="scripted_demo", video=False, auto_label=False,
                                   scene_config=FIXTURE, scene_objects=names)
        recorder.start(data)
        run(recorder, model, data, controller, .5)
        directory = recorder.stop(data, "user_stop")
        metadata = json.loads((directory / "episode.json").read_text())
        info = metadata["scene"]["scene_config"]
        self.assertEqual(info["path"], "assets/photo_scene_test/scene.json")
        self.assertEqual(info, json.loads(json.dumps(scene_summary(FIXTURE))))
        import h5py
        with h5py.File(directory / "episode.h5") as file:
            self.assertIn("scene/photo_test_coffee_carton_pose", file)
            self.assertIn("contacts/hand_photo_test_coffee_carton", file)
        self.assertEqual(episode_model(metadata).nq, model.nq)
        self.assertEqual(asset_warnings(metadata), [])
        info["sha256"] = "0" * 64
        self.assertIn("changed since this episode", asset_warnings(metadata)[0])
        info["path"] = "assets/does_not_exist/scene.json"
        with self.assertRaisesRegex(ValueError, "not in this checkout"):
            episode_model(metadata)


if __name__ == "__main__":
    unittest.main()
