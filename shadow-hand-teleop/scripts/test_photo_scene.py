"""Offline mesh import tests; no SAM inference or headset-validation claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image
import trimesh

from photo_scene import add_photo_scene, load_bundle, normalize_parts, prepare_bundle, triangle_parts


FIXTURE = Path(__file__).resolve().parents[1] / "assets/photo_scene_test/scene.json"


class PhotoSceneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "box.glb"
        trimesh.creation.box(extents=[2, 4, 6]).export(self.source)

    def prepare(self, **kwargs):
        return prepare_bundle(self.source, self.root / "prepared", object_id="test",
            height_m=.12, up_axis="z", **kwargs)

    def write_config(self, config):
        path = self.root / "test.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_uniform_scale_bottom_origin(self):
        path = self.prepare()
        mesh = load_bundle(path)[0]["parts"][0][0]
        np.testing.assert_allclose(mesh.extents, [.04, .08, .12], atol=1e-9)
        np.testing.assert_allclose(mesh.bounds[0], [-.02, -.04, 0], atol=1e-9)
        metadata = json.loads(path.read_text())["objects"][0]
        self.assertFalse(metadata["provenance"]["inference_run_verified"])
        self.assertEqual(metadata["preparation"]["physics_source"], "manual assumptions, not inferred from image")

    def test_source_up_axis_is_explicit(self):
        part = trimesh.creation.box(extents=[2, 4, 6])
        result, _ = normalize_parts([part], .2, "y")
        np.testing.assert_allclose(result[0].extents, [.1, .3, .2], atol=1e-9)

    def test_glb_scene_transforms_and_instances(self):
        scene = trimesh.Scene()
        for x in (0, 3):
            matrix = np.eye(4)
            matrix[0, 3] = x
            scene.add_geometry(trimesh.creation.box(), transform=matrix)
        scene.export(self.source)
        parts = triangle_parts(self.source)
        self.assertEqual(len(parts), 2)
        centres = sorted(part.centroid[0] for part in parts)
        np.testing.assert_allclose(centres, [0, 3], atol=1e-12)

    def test_obj_and_stl_input(self):
        for extension in ("obj", "stl"):
            path = self.root / f"box.{extension}"
            trimesh.creation.box().export(path)
            self.assertEqual(len(triangle_parts(path)), 1)

    def test_gaussian_or_point_cloud_is_rejected(self):
        path = self.root / "points.ply"
        trimesh.points.PointCloud(np.random.default_rng(2).random((20, 3))).export(path)
        with self.assertRaisesRegex(ValueError, "Point clouds"):
            triangle_parts(path)

    def test_destination_is_never_overwritten(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()

    def test_bad_dimensions_and_mass_are_rejected(self):
        for kwargs in ({"height_m": float("nan")}, {"height_m": 0}, {"mass_kg": -1}):
            options = dict(object_id="test", height_m=.12, up_axis="z")
            options.update(kwargs)
            with self.assertRaises(ValueError):
                prepare_bundle(self.source, self.root / "bad", **options)

    def test_bundle_asset_cannot_escape_directory(self):
        path = self.prepare()
        config = json.loads(path.read_text())
        config["objects"][0]["parts"][0]["mesh"] = "../box.glb"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "outside scene bundle"):
            load_bundle(path)

    def test_duplicate_ids_rejected(self):
        path = self.prepare()
        config = json.loads(path.read_text())
        config["objects"].append(copy.deepcopy(config["objects"][0]))
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_bundle(path)

    def test_texture_uv_survives_glb_to_mujoco(self):
        mesh = trimesh.creation.box()
        # Synthetic 2x2 texture tests UV/material plumbing, not reconstruction.
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.linspace(0, 1, len(mesh.vertices) * 2).reshape(-1, 2),
            image=Image.new("RGB", (2, 2), (180, 40, 20)))
        mesh.export(self.source)
        path = self.prepare()
        spec = mujoco.MjSpec()
        add_photo_scene(spec, path)
        model = spec.compile()
        self.assertEqual(model.ntex, 1)
        self.assertEqual(model.nmat, 1)
        self.assertGreater(model.nmeshtexcoord, 0)
        self.assertEqual(model.geom("photo_test_visual_0").contype[0], 0)
        self.assertGreaterEqual(model.geom("photo_test_visual_0").matid[0], 0)

    def test_multiple_boxes_allow_a_manual_cavity(self):
        path = self.prepare()
        config = json.loads(path.read_text())
        config["objects"][0]["collision"] = {"mode": "boxes", "boxes": [
            {"position_m": [-.018, 0, .06], "half_size_m": [.002, .04, .06]},
            {"position_m": [.018, 0, .06], "half_size_m": [.002, .04, .06]}]}
        path.write_text(json.dumps(config))
        spec = mujoco.MjSpec()
        add_photo_scene(spec, path)
        model = spec.compile()
        self.assertAlmostEqual(model.body("photo_test").mass[0], .15)
        self.assertEqual(model.geom("photo_test_collision_0").type[0], mujoco.mjtGeom.mjGEOM_BOX)

    def test_fixed_fixture_has_no_free_joint(self):
        path = self.prepare(fixed=True, collision="box")
        spec = mujoco.MjSpec()
        add_photo_scene(spec, path)
        model = spec.compile()
        self.assertEqual(model.njnt, 0)

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


if __name__ == "__main__":
    unittest.main()
