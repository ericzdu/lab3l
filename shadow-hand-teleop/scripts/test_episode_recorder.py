"""Headless episode recording: scripted targets and a fake tracking stream only."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

from arm_control import SIDES, WristMapper
from episode_recorder import CONTACT_GROUPS, PROJECT_ROOT, SCHEMA, EpisodeRecorder, dataset_layout
from hand_retargeting import FINGER_TARGET_NAMES, synthetic_skeleton
from workspace_scene import CUP_START, SCENE_VERSION, TABLE_TOP, TRAY_CENTER
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
        options = dict(record_dir=self.dir, source="scripted_demo", operator="NT", video=False,
                       auto_label=False)
        options.update(kwargs)
        return EpisodeRecorder(self.model, self.controller, self.hands, **options)

    def test_headless_round_trip_schema(self):
        recorder = self.recorder(auto_label=True)
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
            np.testing.assert_allclose(file["scene/cup_pose"][0, :3], CUP_START, atol=.003)
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
        self.assertEqual(metadata["scene"]["scene_version"], SCENE_VERSION)
        self.assertEqual(metadata["scene"]["table_top"], TABLE_TOP)
        self.assertEqual(metadata["source"], "scripted_demo")
        self.assertEqual(metadata["end_reason"], "user_stop")
        self.assertEqual(metadata["model"]["nq"], self.model.nq)
        self.assertEqual(len(metadata["model"]["actuator_names"]), self.model.nu)
        # Auto-labeled: the scripted demo never touches the cup.
        self.assertEqual(metadata["labels"]["outcome"]["failure_reason"], "never_touched")
        self.assertEqual(recorder.status_text(self.data), f"Saved ep 0001 · FAIL (never_touched)")
        self.assertEqual([e["kind"] for e in events], ["record_start", "pause", "record_stop"])
        # The existing replay path accepts episodes.
        from replay_data import load_demo, validate_model
        loaded = load_demo(directory / "episode.h5")
        validate_model(loaded, self.model)
        self.assertEqual(loaded["_metadata"]["episode_id"], directory.name)
        self.assertEqual(loaded["_labels"]["phase_names"][0], "idle")
        self.assertEqual(len(loaded["_labels"]["phase"]), n)
        np.testing.assert_array_equal(loaded["timestamp"], times)
        script = Path(__file__).with_name("replay_demo.py")
        result = subprocess.run([sys.executable, str(script), str(directory / "episode.h5"), "--headless"],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Replay finished", result.stdout)
        self.assertIn("phase: idle", result.stdout)
        self.assertIn("Outcome: FAIL (never_touched)", result.stdout)
        # Re-labeling via the CLI is idempotent; summarize lists the episode.
        cli = lambda name, *args: subprocess.run(
            [sys.executable, str(Path(__file__).with_name(name)), *map(str, args)],
            capture_output=True, text=True, timeout=120)
        before = (directory / "episode.json").read_text()
        for _ in range(2):
            result = cli("label_episode.py", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FAIL (never_touched)", result.stdout)
            self.assertEqual((directory / "episode.json").read_text(), before)
        result = cli("label_episode.py", self.dir, "--dry-run")
        self.assertIn("Dry run", result.stdout)
        csv_path = Path(self.tmp.name) / "summary.csv"
        result = cli("summarize_episodes.py", self.dir, "--csv", csv_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(directory.name, result.stdout)
        self.assertIn("never_touched", csv_path.read_text())
        # A changed scene layout is refused with a clear error.
        from replay_demo import episode_model
        changed = json.loads(json.dumps(metadata))
        changed["model"]["actuator_names"][0] = "renamed"
        with self.assertRaisesRegex(ValueError, "actuator_names do not match"):
            episode_model(changed)
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

    def test_video_frames_match_alignment_datasets(self):
        import mujoco
        try:
            mujoco.Renderer(self.model, height=120, width=160).close()
        except Exception as exc:
            self.skipTest(f"offscreen rendering unavailable: {exc}")
        import imageio_ffmpeg
        recorder = self.recorder(video=True, video_fps=10, video_size=(160, 120))
        try:
            recorder.start(self.data)
            run(recorder, self.model, self.data, self.controller, 2.0)
            directory = recorder.stop(self.data, "user_stop")
        finally:
            recorder.close()
        metadata = json.loads((directory / "episode.json").read_text())
        video = metadata["video"]
        self.assertEqual(video["status"], "ok")
        self.assertEqual(video["fps"], 10)
        with h5py.File(directory / "episode.h5") as file:
            n = len(file["time/sim"])
            sim = file["time/sim"][...]
            for name in ("operator", "overview"):
                path = directory / f"cam_{name}.mp4"
                self.assertTrue(path.exists(), name)
                frame_time = file[f"video/{name}/frame_time"][...]
                index = file[f"video/{name}/sample_index"][...]
                frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
                self.assertEqual(frames, len(frame_time), name)
                self.assertEqual(frames, video["frames"][name])
                self.assertGreaterEqual(video["dropped_frames"][name], 0)
                expected = 2.0 * 10 - video["dropped_frames"][name]
                self.assertLessEqual(abs(frames - expected), 1, name)
                self.assertTrue(np.all(np.diff(frame_time) > 0))
                self.assertTrue(((index >= 0) & (index < n)).all())
                np.testing.assert_allclose(sim[index], frame_time)
                camera = video["cameras"][name]
                self.assertEqual((camera["width"], camera["height"]), (160, 120))
                self.assertAlmostEqual(camera["fy"], 60 / np.tan(np.deg2rad(camera["fovy"]) / 2))
                self.assertEqual((camera["cx"], camera["cy"]), (80, 60))
        self.assertGreaterEqual(self.model.vis.global_.offwidth, 160)
        self.assertIsNotNone(video["render_ms"]["mean"])
        # Discard while encoding leaves nothing behind.
        recorder = self.recorder(video=True, video_fps=10, video_size=(160, 120))
        try:
            recorder.start(self.data)
            run(recorder, self.model, self.data, self.controller, .5)
            directory = recorder.directory
            recorder.discard()
        finally:
            recorder.close()
        self.assertFalse(directory.exists())

    def test_video_unavailable_keeps_recording_state(self):
        from unittest.mock import patch
        recorder = self.recorder(video=True, video_fps=10, video_size=(160, 120))
        with patch("episode_video.mujoco.Renderer", side_effect=RuntimeError("no GL context")):
            recorder.start(self.data)
        self.assertIn("Video unavailable", recorder.status_text(self.data))
        run(recorder, self.model, self.data, self.controller, .3)
        directory = recorder.stop(self.data, "user_stop")
        metadata = json.loads((directory / "episode.json").read_text())
        self.assertEqual(metadata["video"]["status"], "unavailable: RuntimeError: no GL context")
        self.assertFalse(list(directory.glob("*.mp4")))
        self.assertGreater(metadata["samples"], 20)

    def test_labeler_failure_keeps_recording(self):
        from unittest.mock import patch
        recorder = self.recorder(auto_label=True)
        recorder.start(self.data)
        run(recorder, self.model, self.data, self.controller, .3)
        with patch("episode_labels.label_episode", side_effect=RuntimeError("boom")):
            directory = recorder.stop(self.data, "user_stop")
        metadata = json.loads((directory / "episode.json").read_text())
        self.assertIsNone(metadata["labels"])
        self.assertTrue((directory / "episode.h5").exists())
        with h5py.File(directory / "episode.h5") as file:
            self.assertNotIn("labels", file)

    def test_replay_warns_when_props_moved(self):
        from replay_demo import prop_layout_warning
        current = {"scene_version": SCENE_VERSION, "workspace": True, "table_top": TABLE_TOP,
                   "cup_start": CUP_START.tolist(), "tray_center": TRAY_CENTER.tolist()}
        self.assertIsNone(prop_layout_warning({"scene": current}))
        moved = dict(current, scene_version=SCENE_VERSION + 1, table_top=TABLE_TOP + .1)
        warning = prop_layout_warning({"scene": moved})
        self.assertIn("table_top", warning)
        self.assertIn("NEW positions", warning)
        # An old episode without scene_version is compared as the version-1 layout.
        from workspace_scene import SCENE_V1
        old = prop_layout_warning({"scene": {"workspace": True}})
        same = np.isclose(SCENE_V1["table_top"], TABLE_TOP) and np.allclose(SCENE_V1["tray_center"], TRAY_CENTER)
        self.assertEqual(old is None, bool(same and np.allclose(SCENE_V1["cup_start"], CUP_START)))

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
