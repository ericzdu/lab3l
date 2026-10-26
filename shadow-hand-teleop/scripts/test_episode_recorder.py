"""Headless episode recording: scripted targets and a fake tracking stream only."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

from arm_control import SIDES, WristMapper
from episode_recorder import CONTACT_GROUPS, PROJECT_ROOT, SCHEMA, EpisodeRecorder, dataset_layout
from hand_retargeting import FINGER_TARGET_NAMES, synthetic_skeleton
from view_dual_arms import (RECORDING_REQUIRES, advance_physics, demo_targets, reset_scene,
                            setup, toggle_recording)


def run(recorder, model, data, controller, seconds, samples=None, bridge=None):
    """Mimic the viewer loop: sample only after physics advanced."""
    end = data.time + seconds
    while data.time < end:
        if bridge is not None:
            samples = bridge.snapshot()[0]
        samples = samples or {}
        now = time.monotonic()
        fresh = {side: side in samples and now - samples[side][0] < .5 for side in SIDES}
        demo_targets(controller, data.time)
        controller.command(data, dt=.01)
        advance_physics(model, data, controller)
        recorder.sample(data, samples=samples, fresh=fresh, paused=False, mapper=None)
        recorder.watch(data, fresh)
        recorder.maybe_render(data)
        if recorder.expired(data):
            return recorder.stop(data, "max_duration")
    return None


class RecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.controller, cls.hands = setup("human", workspace=True)

    def setUp(self):
        self.controller.reset(self.data)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "episodes"

    def recorder(self, **kwargs):
        options = dict(record_dir=self.dir, source="scripted_demo", operator="NT", video=False)
        options.update(kwargs)
        return EpisodeRecorder(self.model, self.controller, self.hands, **options)

    def test_headless_round_trip_schema(self):
        recorder = self.recorder()
        recorder.start(self.data)
        run(recorder, self.model, self.data, self.controller, 2.0)
        recorder.event(self.data, "pause")
        directory = recorder.stop(self.data, "user_stop")
        self.assertRegex(directory.name, r"^\d{8}-\d{6}_cup_to_tray_0001$")
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["episode.h5", "episode.json"])
        layout = dataset_layout(self.model)
        with h5py.File(directory / "episode.h5") as file:
            self.assertEqual(file.attrs["schema"], SCHEMA)
            self.assertEqual(file.attrs["schema_version"], 2)
            n = len(file["time/sim"])
            self.assertGreater(n, 150)
            for name, (dtype, shape) in layout.items():
                self.assertEqual(file[name].shape, (n, *shape), name)
                self.assertEqual(file[name].dtype, np.dtype(dtype), name)
            times = file["time/sim"][...]
            self.assertEqual(times[0], 0)
            self.assertTrue(np.all(np.diff(times) > 0))
            self.assertTrue(np.all(np.diff(file["time/wall"][...]) >= 0))
            self.assertEqual(json.loads(file["input/finger_targets"].attrs["names"]),
                             list(FINGER_TARGET_NAMES))
            self.assertEqual(json.loads(file["contacts/hand_cup"].attrs["groups"]), list(CONTACT_GROUPS))
            # Scripted source: input arrays exist but are empty markers.
            self.assertTrue(np.isnan(file["input/finger_skeleton"][...]).all())
            self.assertFalse(file["input/fresh"][...].any())
            self.assertTrue((file["input/packet_seq"][...] == -1).all())
            # Cup rests on the table at the start position.
            self.assertTrue(file["contacts/cup_table"][...][-20:].all())
            np.testing.assert_allclose(file["scene/cup_pose"][0, :3], [.32, -.28, .562], atol=.003)
            np.testing.assert_allclose(file["robot/qpos"][-1], self.data.qpos)
            metadata = json.loads(file.attrs["metadata"])
            events = json.loads(file.attrs["events"])
        sidecar = json.loads((directory / "episode.json").read_text())
        self.assertEqual(metadata, sidecar)
        for key in ("episode_id", "task", "source", "operator", "created_utc", "duration_sim_s",
                    "samples", "end_reason", "scene", "model", "software",
                    "finger_target_names", "video", "labels"):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["samples"], n)
        self.assertEqual(metadata["source"], "scripted_demo")
        self.assertEqual(metadata["end_reason"], "user_stop")
        self.assertEqual(metadata["model"]["nq"], self.model.nq)
        self.assertEqual(len(metadata["model"]["actuator_names"]), self.model.nu)
        self.assertIsNone(metadata["labels"])
        self.assertEqual([e["kind"] for e in events], ["record_start", "pause", "record_stop"])
        # Next episode gets the next counter.
        recorder.start(self.data)
        self.assertTrue(recorder.directory.name.endswith("_0002"))
        recorder.discard()

    def test_fake_bridge_raw_input_and_privacy(self):
        from teleop_shadow_hands import TrackingBridge
        address = "192.0.2.77-SECRET-ROOM"
        fingers = synthetic_skeleton(.4)
        fingers[3, 0, 3] = 1 / 3  # Not exactly representable: tests float32 storage.
        arm = np.repeat(np.eye(4)[None], 27, axis=0)
        arm[:25] = fingers
        arm[26, :3, 3] = [-.25, 0, 0]
        wrist = np.eye(4)
        wrist[:3, 3] = [.1, .2, 1.1]

        class Packet(dict):
            pass

        class FakeStreamer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_latest(self):
                packet = Packet(right_fingers=fingers, right_wrist=wrist[None], right_arm=arm)
                packet._raw = object()  # A new packet every poll.
                return packet

            def cleanup(self):
                pass

        bridge = TrackingBridge(FakeStreamer, address, with_wrist=True)
        bridge.thread.start()
        try:
            deadline = time.monotonic() + 2
            while "right" not in bridge.snapshot()[0] and time.monotonic() < deadline:
                time.sleep(.01)
            entry = bridge.snapshot()[0]["right"]
            self.assertEqual(len(entry), 7)
            np.testing.assert_array_equal(entry[4], fingers)
            np.testing.assert_array_equal(entry[5], arm)
            self.assertIsInstance(entry[6], int)
            recorder = self.recorder(source="live")
            recorder.start(self.data)
            run(recorder, self.model, self.data, self.controller, .5, bridge=bridge)
            recorder.event(self.data, "error", message=f"failed in {Path.home()}/somewhere")
            directory = recorder.stop(self.data, "user_stop")
        finally:
            bridge.close()
        with h5py.File(directory / "episode.h5") as file:
            right = SIDES.index("right")
            stored = file["input/finger_skeleton"][:, right]
            np.testing.assert_array_equal(stored, np.broadcast_to(fingers.astype(np.float32), stored.shape))
            np.testing.assert_array_equal(file["input/arm_skeleton"][:, right][-1], arm.astype(np.float32))
            np.testing.assert_array_equal(file["input/wrist"][-1, right], wrist.astype(np.float32))
            self.assertTrue(np.isnan(file["input/finger_skeleton"][:, 1 - right]).all())
            self.assertTrue(file["input/fresh"][:, right].all())
            seq = file["input/packet_seq"][:, right]
            self.assertTrue((seq >= 0).all() and np.all(np.diff(seq) >= 0))
            self.assertTrue(file["input/new_packet"][:, right].any())
            self.assertTrue(np.isfinite(file["input/forearm_vector"][:, right]).all())
            targets = file["input/finger_targets"][-1, right]
            self.assertTrue(np.isfinite(targets).all())
            events = json.loads(file.attrs["events"])
        self.assertEqual(events[0]["fresh"], {"left": False, "right": True})
        error = [e for e in events if e["kind"] == "error"][0]
        self.assertEqual(error["message"], "failed in ~/somewhere")
        for name in ("episode.h5", "episode.json"):
            content = (directory / name).read_bytes()
            for secret in (address, "SECRET", str(Path.home()), str(PROJECT_ROOT)):
                self.assertNotIn(secret.encode(), content, f"{secret!r} leaked into {name}")

    def test_discard_reset_and_max_duration(self):
        recorder = self.recorder()
        recorder.start(self.data)
        run(recorder, self.model, self.data, self.controller, .2)
        directory = recorder.directory
        recorder.discard()
        self.assertFalse(directory.exists())
        self.assertFalse(recorder.recording)
        self.assertEqual(list(self.dir.iterdir()), [])

        # H saves the episode with end_reason="reset", then resets the scene.
        mapper = WristMapper()
        mapper.calibrate("left", np.eye(4), [0, 0, 0])
        recorder.start(self.data)
        run(recorder, self.model, self.data, self.controller, .2)
        directory = recorder.directory
        reset_scene(self.data, self.controller, mapper, None, recorder)
        self.assertFalse(recorder.recording)
        self.assertEqual(self.data.time, 0)
        self.assertNotIn("left", mapper.origins)
        metadata = json.loads((directory / "episode.json").read_text())
        self.assertEqual(metadata["end_reason"], "reset")
        with h5py.File(directory / "episode.h5") as file:
            kinds = [e["kind"] for e in json.loads(file.attrs["events"])]
        self.assertEqual(kinds[-2:], ["reset", "record_stop"])

        # E toggles; stop with no samples keeps nothing.
        self.assertIsNone(toggle_recording(recorder, self.data))
        self.assertTrue(recorder.recording)
        empty = recorder.directory
        self.assertIsNone(toggle_recording(recorder, self.data))
        self.assertFalse(empty.exists())
        self.assertEqual(toggle_recording(None, self.data), RECORDING_REQUIRES)

        toggle_recording(recorder, self.data, calibrated=False)
        self.assertIn("before calibration", recorder.status_text(self.data))
        recorder.discard()

        short = self.recorder(max_seconds=.3)
        short.start(self.data)
        directory = run(short, self.model, self.data, self.controller, 2.0)
        self.assertIsNotNone(directory)
        metadata = json.loads((directory / "episode.json").read_text())
        self.assertEqual(metadata["end_reason"], "max_duration")
        self.assertAlmostEqual(metadata["duration_sim_s"], .3, delta=.011)
        self.assertRegex(short.status_text(self.data), r"^Saved ep \d{4} · unlabeled$")

    def test_scene_or_argument_errors_are_loud(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            self.recorder(wrist_mode="spherical")
        with self.assertRaisesRegex(ValueError, "task"):
            self.recorder(task="../escape")
        with self.assertRaisesRegex(ValueError, "operator"):
            self.recorder(operator="x" * 33)
        model, *_ = setup("human", workspace=False)
        from episode_recorder import ContactTables
        with self.assertRaisesRegex(ValueError, "cup"):
            ContactTables(model)


if __name__ == "__main__":
    unittest.main()
