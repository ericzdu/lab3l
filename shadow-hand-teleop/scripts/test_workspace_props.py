"""Optional --props: physics on the table, default scene unchanged, replay round trip."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from view_dual_arms import advance_physics, setup
from workspace_scene import PROPS, prop_half_height, prop_start


class PropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.d, cls.c, cls.hands = setup("human", workspace=True, props=list(PROPS))

    def step(self, n):
        for _ in range(n):
            self.c.command(self.d)
            advance_physics(self.m, self.d, self.c)

    def test_default_scene_keeps_its_joints(self):
        # cup_to_tray episodes replay only if the joint list is unchanged.
        default, _, _, _ = setup("human", workspace=True)
        names = lambda m: [m.joint(i).name for i in range(m.njnt)]
        self.assertTrue(all(f"{p}_free" not in names(default) for p in PROPS))
        self.assertEqual(names(self.m), names(default) + [f"{p}_free" for p in PROPS])
        self.assertEqual(self.m.nq, default.nq + 7 * len(PROPS))

    def test_props_rest_on_table_and_reset(self):
        self.c.reset(self.d)
        self.step(600)
        for name in PROPS:
            np.testing.assert_allclose(self.d.body(name).xpos, prop_start(name), atol=.002,
                                       err_msg=name)
            self.assertEqual(self.m.body_gravcomp[self.m.body(name).id], 0)
        self.assertLess(np.max(np.abs(self.d.qvel)), .01)
        q = int(self.m.joint("block_free").qposadr[0])
        self.d.qpos[q] += .05
        self.c.reset(self.d)
        np.testing.assert_allclose(self.d.qpos[q:q + 3], prop_start("block"))

    def test_left_hand_reaches_above_each_prop(self):
        for name in PROPS:
            self.c.reset(self.d)
            goal = prop_start(name) + [0, 0, prop_half_height(PROPS[name]) + .06]
            self.c.set_target("left", goal)
            self.step(250)
            error = np.linalg.norm(self.d.site_xpos[self.c.sites["left"]] - goal)
            self.assertLess(error, .01, name)

    def test_visuals_are_look_only(self):
        """prop_visuals adds no mass, inertia or contacts; collision shapes are hidden."""
        from unittest import mock
        import prop_visuals
        with mock.patch.dict(prop_visuals.VISUALS, clear=True):
            plain, _, _, _ = setup("human", workspace=True, props=list(PROPS))
        for name in PROPS:
            a, b = self.m.body(name).id, plain.body(name).id
            self.assertAlmostEqual(self.m.body_mass[a], plain.body_mass[b], places=12)
            np.testing.assert_allclose(self.m.body_inertia[a], plain.body_inertia[b], rtol=1e-9)
            np.testing.assert_allclose(self.m.body_ipos[a], plain.body_ipos[b], atol=1e-12)
            self.assertEqual(self.m.geom(f"{name}_geom").group[0], prop_visuals.COLLISION_GROUP)
            visual = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == a
                      and self.m.geom(g).name != f"{name}_geom"]
            self.assertTrue(visual, name)
            for g in visual:
                self.assertEqual((self.m.geom_contype[g], self.m.geom_conaffinity[g]), (0, 0))
                self.assertEqual(self.m.geom_group[g], prop_visuals.VISUAL_GROUP)

    def test_lathe_surfaces_are_closed(self):
        """Every edge is shared by exactly two faces, wound in opposite directions."""
        import prop_visuals as pv
        for profile in (pv.apple_profile(.035), pv.bottle_profile(.03, .08),
                        pv.plate_profile(.07, .006), [(.03, 0), (.031, 0), (.031, .01), (.03, .01),
                                                      (.03, 0)]):
            _, faces = pv.lathe(profile)
            edges = {}
            for face in faces:
                for i in range(3):
                    edge = (face[i], face[(i + 1) % 3])
                    edges[edge] = edges.get(edge, 0) + 1
            for (a, b), count in edges.items():
                self.assertEqual(count, 1)
                self.assertIn((b, a), edges)

    def test_unknown_prop_and_props_without_table(self):
        with self.assertRaises(ValueError):
            setup("human", workspace=True, props=["mug"])
        with self.assertRaises(ValueError):
            setup("human", workspace=False, props=["block"])

    def test_record_then_replay_round_trip(self):
        from episode_recorder import EpisodeRecorder
        from replay_demo import episode_model
        from test_episode_recorder import run
        props = ["block", "apple"]
        model, data, controller, hands = setup("human", workspace=True, props=props)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(model, controller, hands, record_dir=Path(tmp),
                                       source="scripted_demo", props=props, video=False,
                                       auto_label=False)
            recorder.start(data)
            run(recorder, model, data, controller, 1.0)
            directory = recorder.stop(data, "user_stop")
            metadata = json.loads((directory / "episode.json").read_text())
        self.assertEqual(metadata["scene"]["props"], props)
        # Replay rebuilds the recorded props, so the joint check passes.
        self.assertEqual(episode_model(metadata).nq, model.nq)
        # Older episodes have no "props" key: replay builds the default scene,
        # whose joints differ, so an episode recorded with props is refused.
        del metadata["scene"]["props"]
        with self.assertRaises(ValueError):
            episode_model(metadata)
        default, _, _, _ = setup("human", workspace=True)
        metadata["model"]["joint_names"] = [default.joint(i).name for i in range(default.njnt)]
        self.assertEqual(episode_model(metadata).nq, default.nq)


if __name__ == "__main__":
    unittest.main()
