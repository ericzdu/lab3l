"""Labeler checks on synthetic episode arrays only (no MuJoCo stepping)."""
import unittest

import numpy as np

from episode_labels import PHASES, LabelConfig, debounce, label_episode
from workspace_scene import CUP_HALF_HEIGHT, CUP_START, TABLE_TOP, TRAY_CENTER

DT = .01
REST = TABLE_TOP + CUP_HALF_HEIGHT
TRAY_REST = REST + .012
UPRIGHT = np.array([1., 0, 0, 0])
ON_SIDE = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])  # 90 deg about x
OFF_TRAY = np.array([.32, -.02, REST])
ABOVE_TRAY = np.array([*TRAY_CENTER, .64])
IN_TRAY = np.array([*TRAY_CENTER, TRAY_REST])
WRIST_HOME = np.array([.30, -.05, .72])


class Script:
    """Piecewise-linear synthetic episode for the right hand (left idle)."""
    def __init__(self):
        self.rows = []
        self.cup, self.wrist, self.quat = CUP_START.copy(), WRIST_HOME.copy(), UPRIGHT

    def segment(self, seconds, cup=None, wrist=None, quat=None, touch=False, table=False,
                tray=False, fresh=True, groups=("thumb", "ff")):
        cup = self.cup if cup is None else np.asarray(cup, float)
        wrist = self.wrist if wrist is None else np.asarray(wrist, float)
        quat = self.quat if quat is None else quat
        steps = int(round(seconds / DT))
        for k in range(1, steps + 1):
            f = k / steps
            self.rows.append(dict(cup=self.cup + f * (cup - self.cup),
                                  wrist=self.wrist + f * (wrist - self.wrist),
                                  quat=quat, touch=touch, table=table, tray=tray, fresh=fresh,
                                  groups=groups))
        self.cup, self.wrist, self.quat = cup, wrist, quat
        return self

    def grasp_to_lift(self, flicker=False):
        near = CUP_START + [0, .02, .06]
        self.segment(.5, table=True)                       # idle
        self.segment(.8, wrist=near, table=True)          # reach
        if flicker:
            self.segment(.06, touch=True, table=True)
            self.segment(.04, touch=False, table=True)
            self.segment(.40, touch=True, table=True)
        else:
            self.segment(.5, touch=True, table=True)      # grasp
        self.segment(.5, cup=CUP_START + [0, 0, .08], wrist=near + [0, 0, .08], touch=True)
        return self

    def transport_and_place(self, release=True, final_quat=None):
        offset = self.wrist - self.cup
        self.segment(1.5, cup=ABOVE_TRAY, wrist=ABOVE_TRAY + offset, touch=True)
        self.segment(.5, cup=IN_TRAY, wrist=IN_TRAY + offset, touch=True, tray=True)
        self.segment(.4, touch=True, tray=True)
        if release:
            self.segment(.2, tray=True, quat=final_quat)
            self.segment(1.0, wrist=self.wrist + [0, 0, .12], tray=True)
        else:
            self.segment(1.0, touch=True, tray=True)
        return self

    def arrays(self):
        n = len(self.rows)
        cup = np.array([row["cup"] for row in self.rows])
        hand_cup = np.zeros((n, 2, 7), bool)
        names = ("thumb", "ff", "mf", "rf", "lf", "palm", "other_hand")
        for i, row in enumerate(self.rows):
            if row["touch"]:
                for group in row["groups"]:
                    hand_cup[i, 1, names.index(group)] = True
        wrist = np.zeros((n, 2, 3))
        wrist[:, 0] = [.30, .25, .72]
        wrist[:, 1] = [row["wrist"] for row in self.rows]
        fresh = np.zeros((n, 2), bool)
        fresh[:, 1] = [row["fresh"] for row in self.rows]
        return {
            "time/sim": np.arange(n) * DT,
            "robot/wrist_pos": wrist,
            "input/fresh": fresh,
            "scene/cup_pose": np.column_stack([cup, [row["quat"] for row in self.rows]]),
            "scene/cup_linvel": np.gradient(cup, DT, axis=0),
            "scene/cup_angvel": np.zeros((n, 3)),
            "contacts/hand_cup": hand_cup,
            "contacts/arm_cup": np.zeros((n, 2), bool),
            "contacts/cup_table": np.array([row["table"] for row in self.rows]),
            "contacts/cup_tray": np.array([row["tray"] for row in self.rows]),
            "contacts/hand_hand": np.zeros(n, bool),
        }


def label(script, source="keyboard", task="cup_to_tray"):
    metadata = {"task": task, "source": source,
                "scene": {"table_top": TABLE_TOP, "tray_center": TRAY_CENTER.tolist()}}
    return label_episode(script.arrays(), metadata)


def phase_sequence(labels):
    return [segment["phase"] for segment in labels["segments"]]


def kinds(labels):
    return [event["kind"] for event in labels["events"]]


class LabelTests(unittest.TestCase):
    def check_structure(self, labels, n):
        self.assertEqual(len(labels["phase"]), n)
        self.assertEqual(labels["phase_names"], list(PHASES))
        segments = labels["segments"]
        self.assertEqual(segments[0]["start_sample"], 0)
        self.assertEqual(segments[-1]["end_sample"], n)
        for before, after in zip(segments, segments[1:]):
            self.assertEqual(before["end_sample"], after["start_sample"])
            self.assertNotEqual(before["phase"], after["phase"])
        for segment in segments:
            codes = labels["phase"][segment["start_sample"]:segment["end_sample"]]
            self.assertTrue((codes == PHASES.index(segment["phase"])).all())
            for key in ("confidence", "fresh_frac", "stability", "margin"):
                self.assertGreaterEqual(segment[key], 0)
                self.assertLessEqual(segment[key], 1)
        self.assertEqual(labels["config"], LabelConfig().__dict__)

    def test_clean_success(self):
        script = Script().grasp_to_lift().transport_and_place()
        labels = label(script)
        self.check_structure(labels, len(script.rows))
        self.assertEqual(phase_sequence(labels),
                         ["idle", "reach", "grasp", "lift", "transport", "place", "release", "retreat"])
        outcome = labels["outcome"]
        self.assertTrue(outcome["success"], outcome)
        self.assertIsNone(outcome["failure_reason"])
        self.assertTrue(all(outcome["criteria"].values()))
        self.assertEqual(outcome["active_side"], "right")
        self.assertFalse(outcome["bimanual"])
        self.assertEqual(outcome["regrasp_count"], 0)
        self.assertGreater(outcome["confidence"], .3)
        for kind in ("first_touch", "grasp_opposition_start", "lift_off", "tray_contact", "release"):
            self.assertIn(kind, kinds(labels))
        self.assertNotIn("drop", kinds(labels))
        grasp = labels["segments"][2]
        self.assertEqual((grasp["side"], grasp["stability"], grasp["margin"]), ("right", 1.0, 1.0))
        self.assertEqual(grasp["fresh_frac"], 1.0)  # keyboard source

    def test_drop_outside_tray(self):
        script = Script().grasp_to_lift()
        offset = script.wrist - script.cup
        script.segment(.6, cup=OFF_TRAY + [0, 0, .08], wrist=OFF_TRAY + [0, 0, .08] + offset, touch=True)
        script.segment(.15, cup=OFF_TRAY)                     # untouched free fall
        script.segment(1.0, table=True)
        labels = label(script)
        self.check_structure(labels, len(script.rows))
        sequence = phase_sequence(labels)
        self.assertIn("transport", sequence)
        self.assertEqual(sequence[-1], "recover")
        self.assertEqual(labels["outcome"]["failure_reason"], "dropped_outside_tray")
        drop = [event for event in labels["events"] if event["kind"] == "drop"][0]
        self.assertEqual(drop["landed"], "table")
        self.assertGreaterEqual(drop["fall"], .02)

    def test_regrasp_after_drop_then_success(self):
        script = Script().grasp_to_lift()
        offset = script.wrist - script.cup
        script.segment(.6, cup=OFF_TRAY + [0, 0, .08], wrist=OFF_TRAY + [0, 0, .08] + offset, touch=True)
        script.segment(.15, cup=OFF_TRAY)
        script.segment(.5, table=True)
        script.segment(.4, wrist=OFF_TRAY + [0, .02, .06], table=True)
        script.segment(.4, touch=True, table=True)
        script.segment(.5, cup=OFF_TRAY + [0, 0, .08], wrist=OFF_TRAY + [0, .02, .14], touch=True)
        script.transport_and_place()
        labels = label(script)
        self.check_structure(labels, len(script.rows))
        sequence = phase_sequence(labels)
        self.assertEqual(sequence[sequence.index("recover") + 1], "grasp")
        self.assertEqual(labels["outcome"]["regrasp_count"], 1)
        self.assertIn("regrasp", kinds(labels))
        self.assertTrue(labels["outcome"]["success"], labels["outcome"])

    def test_placed_but_still_held(self):
        script = Script().grasp_to_lift().transport_and_place(release=False)
        labels = label(script)
        self.assertEqual(phase_sequence(labels)[-1], "place")
        outcome = labels["outcome"]
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["failure_reason"], "not_released")
        self.assertTrue(outcome["criteria"]["in_tray"])

    def test_tipped_in_tray(self):
        script = Script().grasp_to_lift().transport_and_place(final_quat=ON_SIDE)
        labels = label(script)
        self.assertEqual(labels["outcome"]["failure_reason"], "tipped_over")
        self.assertFalse(labels["outcome"]["criteria"]["upright"])
        self.assertIn("tip_over", kinds(labels))

    def test_never_touched(self):
        script = Script().segment(2.0, wrist=WRIST_HOME + [.05, 0, 0], table=True)
        labels = label(script)
        self.assertEqual(phase_sequence(labels), ["idle"])
        self.assertEqual(labels["outcome"]["failure_reason"], "never_touched")
        self.assertIsNone(labels["segments"][0]["side"])

    def test_flicker_within_debounce_lowers_stability_only(self):
        clean = label(Script().grasp_to_lift().transport_and_place())
        flicker = label(Script().grasp_to_lift(flicker=True).transport_and_place())
        self.assertEqual(phase_sequence(flicker), phase_sequence(clean))
        grasp = flicker["segments"][2]
        self.assertEqual(grasp["phase"], "grasp")
        self.assertLess(grasp["stability"], clean["segments"][2]["stability"])
        self.assertLess(grasp["confidence"], clean["segments"][2]["confidence"])
        self.assertTrue(flicker["outcome"]["success"])

    def test_stale_input_lowers_confidence(self):
        fresh = Script().grasp_to_lift().transport_and_place()
        stale = Script().grasp_to_lift().transport_and_place()
        for i, row in enumerate(stale.rows):
            row["fresh"] = i % 2 == 0
        fresh_labels, stale_labels = label(fresh, "live"), label(stale, "live")
        self.assertEqual(phase_sequence(fresh_labels), phase_sequence(stale_labels))
        for a, b in zip(fresh_labels["segments"], stale_labels["segments"]):
            self.assertEqual(a["fresh_frac"], 1.0)
            self.assertAlmostEqual(b["fresh_frac"], .5, delta=.06)
            self.assertLess(b["confidence"], a["confidence"] + 1e-12)
        self.assertLess(stale_labels["outcome"]["confidence"], fresh_labels["outcome"]["confidence"])

    def test_unknown_task_and_short_episode(self):
        script = Script().grasp_to_lift()
        labels = label(script, task="stack_blocks")
        self.assertEqual(labels["outcome"]["outcome"], "not_evaluated")
        self.assertIsNone(labels["outcome"]["success"])
        self.assertIn("lift", phase_sequence(labels))
        short = label(Script().segment(.4, table=True))
        self.assertEqual(short["outcome"]["failure_reason"], "never_touched")
        with self.assertRaises(ValueError):
            label(Script().segment(.01, table=True))

    def test_debounce(self):
        t = np.arange(30) * DT
        raw = np.zeros(30, bool)
        raw[5:8] = True        # 30 ms: absorbed
        raw[12:30] = True      # 180 ms: accepted at its onset
        out = debounce(raw, t, .1)
        self.assertFalse(out[:12].any())
        self.assertTrue(out[12:].all())


if __name__ == "__main__":
    unittest.main()
