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


class TargetLayout(base.Layout):
    """test_episode_labels waypoints with `obj` as the "cup" and `target` as the "tray"."""
    def __init__(self, obj, target="plate", offset=(0, 0)):
        super().__init__(2, base.workspace_scene.TABLE_TOP, object_start(obj),
                         PROPS[target]["xy"])
        self.target = target
        self.rest = self.table_top + object_half_height(obj)
        self.tray_rest = self.rest + 2 * object_half_height(target)
        self.off_tray = np.array([*(self.tray_center + [-.11, .10]), self.rest])
        self.above_tray = np.array([*(self.tray_center + offset), self.tray_rest + .066])
        self.in_tray = np.array([*(self.tray_center + offset), self.tray_rest])
        self.wrist_home = np.array([.30, .05, self.rest + .158])


def PlateLayout(obj):
    return TargetLayout(obj, "plate")


def plate_arrays(script, obj, yaw=0.):
    """Rename the synthetic cup/tray datasets to obj/target and add the target's pose."""
    target = getattr(script.layout, "target", "plate")
    arrays = {}
    for key, value in script.arrays().items():
        if key != "contacts/hand_hand":
            key = key.replace("cup", obj).replace("tray", target)
        arrays[key] = value
    z = base.workspace_scene.TABLE_TOP + object_half_height(target)
    pose = np.r_[script.layout.tray_center, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    arrays[f"scene/{target}_pose"] = np.tile(pose, (len(script.rows), 1))
    return arrays


def place_script(obj, target="plate", offset=(0, 0)):
    return base.Script(TargetLayout(obj, target, offset)).grasp_to_lift().transport_and_place()


def joined(first, second):
    """Two single-object episodes back to back: (script, obj) pairs on the plate.

    While one object is handled the other rests untouched where it is (the
    second at its start, the first where it ended).
    """
    (sa, oa), (sb, ob) = first, second
    a, b = plate_arrays(sa, oa), plate_arrays(sb, ob)
    na, nb = len(a["time/sim"]), len(b["time/sim"])
    hold = lambda value, count: np.repeat(value[None], count, axis=0)
    out = {"time/sim": np.r_[a["time/sim"], a["time/sim"][-1] + base.DT + b["time/sim"]]}
    for key in a.keys() | b.keys():
        if key == "time/sim":
            continue
        own_a, own_b = oa in key.replace("scene/plate", ""), ob in key.replace("scene/plate", "")
        if own_a and not own_b:
            out[key] = np.concatenate([a[key], hold(a[key][-1], nb)])
        elif own_b and not own_a:
            out[key] = np.concatenate([hold(b[key][0], na), b[key]])
        else:
            out[key] = np.concatenate([a[key], b[key]])
    return out


def label_table(arrays):
    return label_episode(arrays, {"task": "set_the_table", "source": "keyboard",
                                  "scene": PlateLayout("apple").metadata_scene()})


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


class StackLabelTests(unittest.TestCase):
    def label(self, script, yaw=0.):
        return label_episode(plate_arrays(script, "block", yaw),
                             {"task": "block_on_big_block", "source": "keyboard",
                              "scene": script.layout.metadata_scene()})

    def test_block_on_big_block_success(self):
        labels = self.label(place_script("block", "big_block"))
        outcome = labels["outcome"]
        self.assertTrue(outcome["success"], outcome)
        self.assertIn("in_big_block", outcome["criteria"])
        self.assertIn("big_block_contact", base.kinds(labels))
        self.assertEqual(base.phase_sequence(labels)[:7],
                         ["idle", "reach", "grasp", "lift", "transport", "place", "release"])

    def test_top_face_turns_with_the_big_block(self):
        # 2 cm off centre in x and y is on the top face (half width 3.5 cm,
        # 1 cm margin); with the base turned 45 degrees the same spot is 2.8 cm
        # along the block's own x axis, off the allowed area.
        script = place_script("block", "big_block", offset=(.02, .02))
        self.assertTrue(self.label(script)["outcome"]["success"])
        outcome = self.label(script, yaw=np.pi / 4)["outcome"]
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["failure_reason"], "not_in_big_block")


class SetTheTableLabelTests(unittest.TestCase):
    def check_cover(self, labels, n):
        self.assertEqual(len(labels["phase"]), n)
        segments = labels["segments"]
        self.assertEqual(segments[0]["start_sample"], 0)
        self.assertEqual(segments[-1]["end_sample"], n)
        for before, after in zip(segments, segments[1:]):
            self.assertEqual(before["end_sample"], after["start_sample"])

    def test_both_on_the_plate_in_either_order(self):
        for order in (("apple", "block"), ("block", "apple")):
            with self.subTest(order=order):
                arrays = joined(*[(place_script(obj), obj) for obj in order])
                labels = label_table(arrays)
                outcome = labels["outcome"]
                self.assertTrue(outcome["success"], outcome)
                self.assertEqual(outcome["order"], [f"{obj}_to_plate" for obj in order])
                for obj in ("apple", "block"):
                    self.assertTrue(outcome["criteria"][f"{obj}_in_plate"])
                self.check_cover(labels, len(arrays["time/sim"]))
                # Each object gets its own full pick-and-place, in the order done.
                objects = [segment["object"] for segment in labels["segments"]]
                self.assertEqual(objects[0], order[0])
                self.assertEqual(objects[-1], order[1])
                for obj in order:
                    phases = [s["phase"] for s in labels["segments"] if s["object"] == obj]
                    self.assertTrue(base_subsequence(["reach", "grasp", "lift", "transport",
                                                      "place", "release"], phases), (obj, phases))

    def test_missing_step_fails(self):
        idle_block = base.Script(TargetLayout("block")).segment(2.0, table=True)
        labels = label_table(joined((place_script("apple"), "apple"), (idle_block, "block")))
        outcome = labels["outcome"]
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["failure_reason"], "block_never_touched")
        self.assertTrue(outcome["steps"]["apple_to_plate"]["success"])

    def test_first_object_knocked_off_later_fails(self):
        arrays = joined((place_script("apple"), "apple"), (place_script("block"), "block"))
        late = arrays["time/sim"] >= arrays["time/sim"][-1] - 1.5
        arrays["scene/apple_pose"][late, :2] += [.12, 0]
        arrays["contacts/apple_plate"][late] = False
        arrays["contacts/apple_table"][late] = True
        outcome = label_table(arrays)["outcome"]
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["failure_reason"], "apple_not_in_plate")
        self.assertTrue(outcome["steps"]["block_to_plate"]["success"])

    def test_needed_covers_both_steps(self):
        keys = needed({"task": "set_the_table"})
        for key in ("scene/apple_pose", "scene/block_pose", "contacts/apple_plate",
                    "contacts/block_plate", "scene/plate_pose"):
            self.assertIn(key, keys)
        self.assertEqual(len(keys), len(set(keys)))


def base_subsequence(needle, haystack):
    it = iter(haystack)
    return all(item in it for item in needle)


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
        args = parse_args_for(["--wrist", "human", "--workspace", "--task", "set_the_table"])
        self.assertEqual(sorted(args.props), ["apple", "block", "plate"])
        args = parse_args_for(["--wrist", "human", "--workspace", "--task", "block_on_big_block"])
        self.assertEqual(sorted(args.props), ["big_block", "block"])

    def test_set_the_table_waits_for_every_object(self):
        from episode_recorder import EpisodeRecorder
        props = task_props("set_the_table")
        model, data, controller, hands = setup("human", workspace=True, props=props)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(model, controller, hands, record_dir=Path(tmp),
                                       task="set_the_table", props=props, video=False)
            self.assertEqual(recorder.task_objects(), ["apple", "plate", "block"])
            q = int(model.joint("block_free").qposadr[0])
            data.qpos[q] += .05
            mujoco_forward(model, data)
            self.assertEqual(toggle_recording(recorder, data), not_at_start("block"))
            recorder.close()

    def test_each_task_is_reachable_by_one_hand(self):
        """The tray is a right-hand target; the plate and big block are left-hand ones.

        The palm-down hand holds things ahead of its wrist site (fingers +x,
        thumb toward the body's midline), so the wrist goal is the object or
        plate position minus that grasp offset, with the palm 2 cm above the
        object's top (at its start, and resting on the target).
        """
        grasp = {"left": np.array([.10, -.04]), "right": np.array([.10, .04])}
        model, data, controller, _ = setup("human", workspace=True, props=list(PROPS))
        hand = {"tray": "right", "plate": "left", "big_block": "left"}

        def error(side, goal):
            controller.reset(data)
            controller.set_target(side, goal)
            for _ in range(250):
                controller.command(data)
                advance_physics(model, data, controller)
            return np.linalg.norm(data.site_xpos[controller.sites[side]] - goal)

        for task, spec in TASKS.items():
            if "steps" in spec or spec["object"] == "cup":
                continue   # steps are checked as single tasks; scripted_cup_to_tray covers the cup
            side = hand[spec["target"]]
            obj, target = spec["object"], spec["target"]
            height = 2 * object_half_height(obj) + .02
            table = base.workspace_scene.TABLE_TOP
            for xy, z in ((object_start(obj)[:2], table + height),
                          (PROPS[target]["xy"], table + 2 * object_half_height(target) + height)):
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
    """Like test_scripted_cup_to_tray: each left-hand prop task is solvable within
    the operator's limits and labels as SUCCESS end to end (record, save, label)."""
    def test_scripted_prop_tasks_succeed(self):
        import contextlib
        import io
        from scripted_prop_tasks import SCRIPTED_TASKS, SIDE, run
        from test_scripted_cup_to_tray import is_subsequence, phase_order
        for task in SCRIPTED_TASKS:
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
