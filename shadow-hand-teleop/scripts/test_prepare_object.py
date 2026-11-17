"""prepare_object.py: mesh -> object folder (units, axes, collision, provenance)."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image
import trimesh

import object_assets
from prepare_object import footprint_angle, normalize_parts, prepare, triangle_parts


def u_shape():
    """Concave test solid (a U seen from the front): one hull would fill the gap.

    Three touching boxes; CoACD's preprocessing remeshes them into one solid.
    """
    move = trimesh.transformations.translation_matrix
    return trimesh.util.concatenate([
        trimesh.creation.box(extents=[.2, .1, .04], transform=move([0, 0, .02])),
        trimesh.creation.box(extents=[.04, .1, .12], transform=move([-.08, 0, .10])),
        trimesh.creation.box(extents=[.04, .1, .12], transform=move([.08, 0, .10]))])


class PrepareObjectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "box.glb"
        trimesh.creation.box(extents=[2, 4, 6]).export(self.source)

    def prepare(self, name="out", **kwargs):
        options = dict(object_id="test", up_axis="z", height_m=.12, mass_kg=.2,
                       collision="convex")
        options.update(kwargs)
        return prepare(self.source, self.root / name, **options)

    def test_uniform_scale_bottom_origin(self):
        config = object_assets.load(self.prepare().parent)
        np.testing.assert_allclose(config["size_m"], [.04, .08, .12], atol=1e-6)
        vertices = trimesh.load(self.root / "out/visual_0.obj", force="mesh", process=False).vertices
        np.testing.assert_allclose(vertices.min(axis=0), [-.02, -.04, 0], atol=1e-9)
        self.assertEqual(config["values"], {"size": "assumed", "mass": "assumed", "friction": "assumed"})
        self.assertEqual(len(config["source"]["sha256"]), 64)

    def test_width_scale_and_measured_flags(self):
        config = object_assets.load(self.prepare(height_m=None, width_m=.2, measured=["width", "mass"]).parent)
        np.testing.assert_allclose(config["size_m"], [.1, .2, .3], atol=1e-6)
        self.assertEqual(config["values"]["size"], "measured")
        self.assertEqual(config["values"]["mass"], "measured")
        config = object_assets.load(self.prepare("unit", height_m=None, scale=.01).parent)
        np.testing.assert_allclose(config["size_m"], [.02, .04, .06], atol=1e-6)
        with self.assertRaises(ValueError):
            self.prepare("both", width_m=.2)          # height and width at once

    def test_source_up_axis_is_explicit(self):
        result, _ = normalize_parts([trimesh.creation.box(extents=[2, 4, 6])], .2, "y")
        np.testing.assert_allclose(result[0].extents, [.1, .3, .2], atol=1e-9)

    def test_align_footprint_squares_a_rotated_box(self):
        box = trimesh.creation.box(extents=[.02, .04, .03])
        box.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(30), [0, 0, 1]))
        self.assertAlmostEqual(abs(footprint_angle([box])), 30, places=3)
        parts, _ = normalize_parts([box], .03, "z", align_footprint=True)
        np.testing.assert_allclose(sorted(parts[0].extents[:2]), [.02, .04], atol=1e-9)

    def test_glb_scene_transforms_and_instances(self):
        scene = trimesh.Scene()
        for x in (0, 3):
            scene.add_geometry(trimesh.creation.box(), transform=trimesh.transformations.translation_matrix([x, 0, 0]))
        scene.export(self.source)
        centres = sorted(part.centroid[0] for part in triangle_parts(self.source))
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

    def test_bad_inputs_are_rejected(self):
        for kwargs in ({"height_m": float("nan")}, {"height_m": 0}, {"mass_kg": -1},
                       {"object_id": "1bad"}, {"friction": (1, -1, 0)}, {"measured": ["colour"]}):
            with self.assertRaises(ValueError):
                self.prepare("bad", **kwargs)
        self.assertFalse((self.root / "bad").exists())

    def test_coacd_keeps_a_concave_opening(self):
        u_shape().export(self.source)
        config = object_assets.load(self.prepare(height_m=.16, collision="coacd").parent)
        hulls = config["collision"]["hulls"]
        self.assertGreaterEqual(len(hulls), 2)
        # The pieces' total volume stays near the U's own volume, well below
        # its bounding box (one convex hull would fill the gap between arms).
        size = np.asarray(config["size_m"])
        self.assertLess(sum(h["volume_m3"] for h in hulls), .8 * np.prod(size))
        for hull in hulls:
            self.assertTrue((self.root / "out" / hull["mesh"]).is_file())

    def test_box_collision_is_a_primitive(self):
        config = object_assets.load(self.prepare(collision="box").parent)
        self.assertEqual(config["collision"]["mode"], "boxes")
        np.testing.assert_allclose(config["collision"]["boxes"][0]["half_size_m"], [.02, .04, .06], atol=1e-6)
        self.assertFalse((self.root / "out/collision").exists())

    def test_texture_uv_survives_into_mujoco(self):
        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.linspace(0, 1, len(mesh.vertices) * 2).reshape(-1, 2),
            image=Image.new("RGB", (4, 4), (180, 40, 20)))
        mesh.export(self.source)
        folder = self.prepare(texture_size=2).parent
        self.assertEqual(Image.open(folder / "visual_0.png").size, (2, 2))
        spec = mujoco.MjSpec()
        object_assets.add_body(spec, folder, "thing", [0, 0, .1])
        model = spec.compile()
        self.assertEqual(model.ntex, 1)
        self.assertGreater(model.nmeshtexcoord, 0)
        self.assertGreaterEqual(model.geom("thing_visual_0").matid[0], 0)
        self.assertEqual(model.geom("thing_visual_0").contype[0], 0)

    def test_keep_source_and_provenance(self):
        folder = self.prepare(keep_source=True, source_kind="sam3d_export", license="mine",
                              notes=["hello"]).parent
        config = json.loads((folder / "object.json").read_text())
        self.assertEqual(config["source"]["kept_file"], "source/box.glb")
        self.assertTrue((folder / "source/box.glb").is_file())
        self.assertEqual(config["source"]["kind"], "sam3d_export")
        self.assertIn("hello", config["notes"])


if __name__ == "__main__":
    unittest.main()
