"""Scored prop tasks (workspace_scene.TASKS): labels, recording and the record gate."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import test_episode_labels as base
from episode_labels import label_episode, load_episode, needed
from view_dual_arms import advance_physics, not_at_start, parse_args, setup, toggle_recording
from workspace_scene import PROPS, TASKS, object_half_height, object_start, task_props


class PlateLayout(base.Layout):
    """test_episode_labels waypoints with `obj` as the "cup" and the plate as the "tray"."""
    def __init__(self, obj):
        super().__init__(2, base.workspace_scene.TABLE_TOP, object_start(obj),
                         PROPS["plate"]["xy"])
        self.rest = self.table_top + object_half_height(obj)
        self.tray_rest = self.rest + 2 * PROPS["plate"]["size"][1]
        self.off_tray = np.array([*(self.tray_center + [-.11, .10]), self.rest])
        self.above_tray = np.array([*self.tray_center, self.tray_rest + .066])
        self.in_tray = np.array([*self.tray_center, self.tray_rest])
        self.wrist_home = np.array([.30, .05, self.rest + .158])


def plate_arrays(script, obj):
    """Rename the synthetic cup/tray datasets to obj/plate and add the plate's pose."""
    arrays = {}
    for key, value in script.arrays().items():
        if key != "contacts/hand_hand":
            key = key.replace("cup", obj).replace("tray", "plate")
        arrays[key] = value
    plate = np.r_[script.layout.tray_center, base.workspace_scene.TABLE_TOP + .006, 1, 0, 0, 0]
    arrays["scene/plate_pose"] = np.tile(plate, (len(script.rows), 1))
    return arrays


def label_plate(script, obj, task):
    return label_episode(plate_arrays(script, obj), {"task": task, "source": "keyboard",
                                                     "scene": script.layout.metadata_scene()})


class TaskLabelTests(unittest.TestCase):
    def test_needed_keys_follow_the_task(self):
        self.assertIn("contacts/hand_cup", needed({"task": "cup_to_tray"}))
        keys = needed({"task": "apple_to_plate"})
        for key in ("scene/apple_pose", "contacts/hand_apple", "contacts/apple_table",
                    "contacts/apple_plate", "scene/plate_pose", "contacts/hand_hand"):
            self.assertIn(key, keys)
        self.assertFalse(any("cup" in key or "tray" in key for key in keys))

    def test_apple_to_plate_success(self):
        layout = PlateLayout("apple")
        script = base.Script(layout).grasp_to_lift().transport_and_place()
        labels = label_plate(script, "apple", "apple_to_plate")
        outcome = labels["outcome"]
        self.assertTrue(outcome["success"], outcome)
        self.assertIn("in_plate", outcome["criteria"])
        self.assertNotIn("upright", outcome["criteria"])   # an apple has no "up"
        self.assertEqual(base.phase_sequence(labels)[:7],
                         ["idle", "reach", "grasp", "lift", "transport", "place", "release"])
        self.assertIn("plate_contact", base.kinds(labels))
        self.assertNotIn("tray_contact", base.kinds(labels))

    def test_block_on_its_side_still_succeeds(self):
        # cup_to_tray would fail this as tipped_over; a block may land on any face.
        layout = PlateLayout("block")
        script = base.Script(layout).grasp_to_lift().transport_and_place(final_quat=base.ON_SIDE)
        self.assertTrue(label_plate(script, "block", "block_to_plate")["outcome"]["success"])

    def test_not_released_on_plate(self):
        layout = PlateLayout("apple")
        script = base.Script(layout).grasp_to_lift().transport_and_place(release=False)
        outcome = label_plate(script, "apple", "apple_to_plate")["outcome"]
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["failure_reason"], "not_released")

    def test_left_on_table_names_the_plate(self):
        layout = PlateLayout("apple")
        script = base.Script(layout).grasp_to_lift()
        script.segment(1.0, cup=layout.off_tray, touch=True)
        script.segment(.5, cup=layout.off_tray, table=True)
        script.segment(1.0, wrist=script.wrist + [0, 0, .1], table=True)
        outcome = label_plate(script, "apple", "apple_to_plate")["outcome"]
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["failure_reason"], "not_in_plate")


class TaskSceneTests(unittest.TestCase):
    def test_task_adds_its_props(self):
        self.assertEqual(task_props("cup_to_tray"), [])
        for task, spec in TASKS.items():
            self.assertTrue(set(task_props(task)) <= PROPS.keys())
        args = parse_args_for(["--wrist", "human", "--workspace", "--task", "apple_to_plate"])
        self.assertEqual(sorted(args.props), ["apple", "plate"])
        args = parse_args_for(["--wrist", "human", "--workspace", "--props", "all",
                               "--task", "block_to_plate"])
        self.assertEqual(args.props, list(PROPS))

    def test_each_task_is_reachable_by_one_hand(self):
        """The tray is a right-hand target and the plate a left-hand one.

        The palm-down hand holds things ahead of its wrist site (fingers +x,
        thumb toward the body's midline), so the wrist goal is the object or
        plate position minus that grasp offset, with the palm 2 cm above the
        object's top (at its start, and resting on the target).
        """
        grasp = {"left": np.array([.10, -.04]), "right": np.array([.10, .04])}
        model, data, controller, _ = setup("human", workspace=True, props=list(PROPS))
        hand = {"tray": "right", "plate": "left"}

        def error(side, goal):
            controller.reset(data)
            controller.set_target(side, goal)
            for _ in range(250):
                controller.command(data)
                advance_physics(model, data, controller)
            return np.linalg.norm(data.site_xpos[controller.sites[side]] - goal)

        for task, spec in TASKS.items():
            if spec["object"] == "cup":
                continue   # scripted_cup_to_tray covers the cup's palm-inward grasp
            side = hand[spec["target"]]
            obj, plate = spec["object"], PROPS[spec["target"]]
            height = 2 * object_half_height(obj) + .02
            table = base.workspace_scene.TABLE_TOP
            for xy, z in ((object_start(obj)[:2], table + height),
                          (plate["xy"], table + 2 * plate["size"][1] + height)):
                goal = np.r_[np.asarray(xy) - grasp[side], z]
                self.assertLess(error(side, goal), .01, (task, side, goal))

    def test_record_gate_and_prop_datasets(self):
        from episode_recorder import EpisodeRecorder
        from test_episode_recorder import run
        props = ["apple", "plate"]
        model, data, controller, hands = setup("human", workspace=True, props=props)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(model, controller, hands, record_dir=Path(tmp),
                                       task="apple_to_plate", source="scripted_demo",
                                       props=props, video=False)
            # Moving the apple blocks recording with an apple-specific message.
            q = int(model.joint("apple_free").qposadr[0])
            data.qpos[q] += .05
            mujoco_forward(model, data)
            self.assertEqual(toggle_recording(recorder, data), not_at_start("apple"))
            self.assertFalse(recorder.recording)
            controller.reset(data)
            self.assertIsNone(toggle_recording(recorder, data))
            run(recorder, model, data, controller, 1.5)
            directory = recorder.stop(data, "user_stop")
            with h5py.File(directory / "episode.h5") as file:
                for key in ("scene/apple_pose", "contacts/hand_apple", "contacts/apple_plate",
                            "contacts/plate_table", "scene/cup_pose", "contacts/hand_cup"):
                    self.assertIn(key, file)
                np.testing.assert_allclose(file["scene/apple_pose"][0, :3],
                                           object_start("apple"), atol=.003)
                self.assertTrue(file["contacts/apple_table"][...].all())  # resting the whole time
                self.assertTrue(file["contacts/plate_table"][...].all())
            arrays, metadata = load_episode(directory)
            self.assertIn("scene/plate_pose", arrays)
            # Auto-labeled on save with the task-aware labeler.
            outcome = metadata["labels"]["outcome"]
            self.assertIs(outcome["success"], False)
            self.assertEqual(outcome["failure_reason"], "never_touched")

    def test_cup_task_gate_unchanged(self):
        from view_dual_arms import CUP_NOT_AT_START
        self.assertEqual(not_at_start("cup"), CUP_NOT_AT_START)
        self.assertIn("apple", not_at_start("apple"))


def mujoco_forward(model, data):
    import mujoco
    mujoco.mj_forward(model, data)


def parse_args_for(argv):
    import sys
    from unittest import mock
    with mock.patch.object(sys, "argv", ["view_dual_arms.py", *argv]):
        return parse_args()



class ScriptedPlateTests(unittest.TestCase):
    """Like test_scripted_cup_to_tray: each plate task is solvable within the
    operator's limits and labels as SUCCESS end to end (record, save, label)."""
    def test_scripted_plate_tasks_succeed(self):
        import contextlib
        import io
        from scripted_prop_to_plate import PLATE_TASKS, SIDE, run
        from test_scripted_cup_to_tray import is_subsequence, phase_order
        for task in PLATE_TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder, \
                    contextlib.redirect_stdout(io.StringIO()):
                directory, labels, script = run(task, folder)
                outcome = labels["outcome"]
                self.assertTrue(outcome["success"], outcome)
                self.assertEqual(outcome["active_side"], SIDE)
                self.assertTrue(is_subsequence(["reach", "grasp", "lift", "transport", "release"],
                                               phase_order(labels)), phase_order(labels))
                self.assertLess(script.max_reach_error, 1e-9)   # never clipped


if __name__ == "__main__":
    unittest.main()
