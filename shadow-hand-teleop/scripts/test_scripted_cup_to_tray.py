"""The scripted right-hand cup-to-tray run labels as SUCCESS (video off, ~3 s)."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from check_reach import MIN_CLEARANCE, limit_clearance
from episode_labels import LabelConfig
from scripted_cup_to_tray import SIDE, run, waypoints


def phase_order(labels):
    phases = []
    for segment in labels["segments"]:
        if not phases or phases[-1] != segment["phase"]:
            phases.append(segment["phase"])
    return phases


def is_subsequence(wanted, sequence):
    remaining = iter(sequence)
    return all(phase in remaining for phase in wanted)


class ScriptedCupToTrayTests(unittest.TestCase):
    def test_scripted_episode_is_success(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            directory, labels, script = run(folder, video=False)
            self.assertTrue((Path(directory) / "episode.h5").exists())
        outcome = labels["outcome"]
        self.assertEqual(labels["config"], LabelConfig().__dict__)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["active_side"], SIDE)
        self.assertTrue(is_subsequence(
            ["reach", "grasp", "lift", "transport", "place", "release"], phase_order(labels)),
            phase_order(labels))
        # Within the operator's limits: no requested target was ever clipped.
        self.assertEqual(script.max_reach_error, 0)
        # Grasp and placement keep >= 3 cm from every clip/reach limit.
        points = waypoints(script.controller)
        for name in ("grasp", "lift", "place"):
            self.assertGreaterEqual(limit_clearance(script.controller, SIDE, points[name]),
                                    MIN_CLEARANCE, name)


if __name__ == "__main__":
    unittest.main()
