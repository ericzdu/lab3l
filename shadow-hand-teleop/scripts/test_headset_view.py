"""Headset view and the bridge's streamer hook, with fake streamers (no headset, no network)."""
import contextlib
import io
import sys
import time
import unittest
from unittest import mock

import mujoco
import numpy as np

from hand_retargeting import synthetic_skeleton
from headset_view import HeadsetView
from teleop_shadow_hands import TrackingBridge


class Packet(dict):
    pass


def tracking_packet():
    packet = Packet(left_fingers=synthetic_skeleton(), left_wrist=np.eye(4)[None])
    packet._raw = object()
    return packet


class FakeStreamer:
    """Records the outbound-video calls HeadsetView makes."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.frames = []
        self.connected = False
        self.packet = tracking_packet()
        FakeStreamer.instances.append(self)

    def get_latest(self):
        return self.packet

    def configure_video(self, **kwargs):
        self.calls.append(("configure_video", kwargs))

    def start_webrtc(self, **kwargs):
        self.calls.append(("start_webrtc", kwargs))
        return True

    def update_frame(self, frame):
        self.frames.append(frame)

    def is_connected(self):
        return self.connected

    def cleanup(self):
        pass


def wait_for(condition, seconds=2.):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(.005)
    return condition()


class BridgeHookTests(unittest.TestCase):
    def run_bridge(self, **kwargs):
        bridge = TrackingBridge(FakeStreamer, "test", with_wrist=True, **kwargs)
        bridge.thread.start()
        try:
            self.assertTrue(wait_for(lambda: "left" in bridge.snapshot()[0]))
            time.sleep(.05)
        finally:
            bridge.close()
        return bridge

    def test_on_streamer_called_once_with_the_streamer(self):
        seen = []
        bridge = self.run_bridge(on_streamer=seen.append)
        self.assertEqual(seen, [bridge.streamer])
        self.assertIsInstance(seen[0], FakeStreamer)
        self.assertIsNone(bridge.hook_error)
        self.assertIsNone(bridge.snapshot()[1])

    def test_hook_failure_keeps_tracking(self):
        def broken(streamer):
            raise RuntimeError("no video")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bridge = self.run_bridge(on_streamer=broken)
        self.assertEqual(bridge.hook_error, "RuntimeError: no video")
        self.assertIn("hand tracking continues", out.getvalue())
        samples, error = bridge.snapshot()
        self.assertIsNone(error)
        self.assertEqual(len(samples["left"]), 7)   # tuple layout unchanged

    def test_no_hook_is_unchanged(self):
        bridge = self.run_bridge()
        self.assertIsNone(bridge.on_streamer)
        self.assertIsNone(bridge.hook_error)
        self.assertEqual(len(bridge.snapshot()[0]["left"]), 7)


def renderer_available(model):
    try:
        mujoco.Renderer(model, height=32, width=32).close()
        return True
    except Exception:
        return False


class Clock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        return self.t


class HeadsetViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from view_dual_arms import setup
        cls.model, cls.data, cls.controller, _ = setup("human", workspace=True)
        mujoco.mj_forward(cls.model, cls.data)
        cls.can_render = renderer_available(cls.model)

    def view(self, **kwargs):
        view = HeadsetView(self.model, **kwargs)
        self.addCleanup(view.close)
        return view

    def attached(self, **kwargs):
        view = self.view(**kwargs)
        streamer = FakeStreamer()
        streamer.connected = True
        view.attach(streamer)
        self.assertTrue(wait_for(lambda: ("start_webrtc", {"blocking": False}) in streamer.calls))
        return view, streamer

    def viewer_cam(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.3, -.1, .7]
        camera.distance, camera.azimuth, camera.elevation = 1.1, 20., -40.
        return camera

    def test_attach_configures_video_then_starts_webrtc(self):
        view, streamer = self.attached(size=(960, 540), fps=20)
        self.assertEqual(streamer.calls[0], ("configure_video",
                         {"device": None, "size": "960x540", "fps": 20, "stereo": False}))
        self.assertEqual(streamer.calls[1][0], "start_webrtc")
        self.assertEqual(view.status, "waiting for headset")
        self.assertGreaterEqual(self.model.vis.global_.offwidth, 960)
        self.assertGreaterEqual(self.model.vis.global_.offheight, 540)

    def test_frames_are_new_bgr_arrays_and_rate_limited(self):
        if not self.can_render:
            self.skipTest("mujoco.Renderer unavailable")
        clock = Clock()
        view, streamer = self.attached(size=(320, 240), fps=30, clock=clock, budget=10.)
        camera = self.viewer_cam()
        for _ in range(100):            # tight loop, no wall time passes
            view.maybe_send(self.data, camera)
        self.assertEqual(len(streamer.frames), 1)
        for _ in range(3):
            clock.t += 1 / 30 + 1e-6
            view.maybe_send(self.data, camera)
        self.assertEqual(len(streamer.frames), 4)
        self.assertEqual(view.status, "streaming")
        for frame in streamer.frames:
            self.assertEqual((frame.shape, frame.dtype), ((240, 320, 3), np.uint8))
            self.assertTrue(frame.flags.c_contiguous)
        self.assertEqual(len({id(frame) for frame in streamer.frames}), 4)
        # BGR: channel order is the renderer's RGB reversed.
        rgb = view.render(self.data, camera)
        np.testing.assert_array_equal(streamer.frames[-1], rgb[..., ::-1])
        # Real clock: 100 calls in a tight loop send about 0-2 frames (more
        # only if the loop itself took longer than a frame period).
        view2, streamer2 = self.attached(size=(320, 240), fps=30)
        started = time.monotonic()
        for _ in range(100):
            view2.maybe_send(self.data, camera)
        elapsed = time.monotonic() - started
        self.assertLessEqual(len(streamer2.frames), 2 + int(elapsed * 30), f"{elapsed:.3f} s")

    def test_over_budget_skips_a_frame(self):
        if not self.can_render:
            self.skipTest("mujoco.Renderer unavailable")
        clock = Clock()
        view, streamer = self.attached(size=(320, 240), fps=30, clock=clock, budget=0.)
        view.maybe_send(self.data, None)
        clock.t += 1 / 30 + 1e-6
        view.maybe_send(self.data, None)   # skipped: previous frame was over budget
        clock.t += 1 / 30 + 1e-6
        view.maybe_send(self.data, None)
        self.assertEqual((len(streamer.frames), view.skipped), (2, 2))

    def test_follow_copies_viewer_camera_without_mutating_it(self):
        view = self.view(size=(320, 240))
        camera = self.viewer_cam()
        before = (camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation)
        view.follow(camera)
        np.testing.assert_allclose(view.camera.lookat, before[0])
        self.assertEqual((view.camera.distance, view.camera.azimuth, view.camera.elevation), before[1:])
        self.assertIsNot(view.camera, camera)
        camera.azimuth, camera.distance = 75., 1.6
        view.follow(camera)
        self.assertEqual((view.camera.azimuth, view.camera.distance), (75., 1.6))
        view.camera.lookat[:] = 0
        np.testing.assert_allclose(camera.lookat, before[0])
        if self.can_render:
            first = view.render(self.data, camera).copy()
            camera.azimuth = 160.
            self.assertFalse(np.array_equal(view.render(self.data, camera), first))
            self.assertEqual(camera.azimuth, 160.)

    def test_fixed_cameras_ignore_viewer(self):
        from view_dual_arms import WORKSPACE_OVERVIEW
        view = self.view(size=(320, 240), camera="overview")
        view.render(self.data, self.viewer_cam()) if self.can_render else None
        self.assertEqual(view.camera.azimuth, WORKSPACE_OVERVIEW["azimuth"])

    def test_waiting_until_headset_connects(self):
        clock = Clock()
        view = self.view(size=(320, 240), clock=clock)
        streamer = FakeStreamer()
        view.attach(streamer)
        for _ in range(3):
            clock.t += 1
            view.maybe_send(self.data, None)
        self.assertEqual(streamer.frames, [])
        self.assertEqual(view.status, "waiting for headset")
        clock.t += 31
        view.maybe_send(self.data, None)
        self.assertTrue(any("no headset connected" in m for m in view.pop_messages()))

    def test_errors_set_status_and_never_raise(self):
        class BrokenConfigure(FakeStreamer):
            def configure_video(self, **kwargs):
                raise RuntimeError("no codec")
        view = self.view(size=(320, 240))
        view.attach(BrokenConfigure())
        self.assertEqual(view.status, "error: RuntimeError: no codec")
        view.maybe_send(self.data, None)   # no-op after an error

        class BrokenStart(FakeStreamer):
            def start_webrtc(self, **kwargs):
                raise OSError("port in use")
        view = self.view(size=(320, 240))
        view.attach(BrokenStart())
        self.assertTrue(wait_for(lambda: view.status.startswith("error")))
        self.assertIn("port in use", view.status)
        self.assertTrue(any("teleoperation continues" in m for m in view.pop_messages()))

        if not self.can_render:
            return
        class BrokenUpdate(FakeStreamer):
            def update_frame(self, frame):
                raise ValueError("bad frame")
        clock = Clock()
        view = self.view(size=(320, 240), clock=clock)
        streamer = BrokenUpdate()
        streamer.connected = True
        view.attach(streamer)
        view.maybe_send(self.data, None)
        self.assertEqual(view.status, "error: ValueError: bad frame")

    def test_overlay_draws_bottom_left_only(self):
        from headset_view import draw_overlay, plain
        frame = np.full((720, 1280, 3), 200, np.uint8)
        draw_overlay(frame, ["L: LIVE \u00b7 FOLLOW   R: WAIT \u00b7 press C", "\u25cf REC 00:12",
                             "Last: ep 0003 \u00b7 SUCCESS"])
        changed = np.any(frame != 200, axis=2)
        rows, cols = np.nonzero(changed)
        self.assertGreater(rows.min(), 720 * .7)       # bottom strip only
        self.assertLess(cols.max(), 1280 * .6)         # left side only
        # Red record dot (BGR) is drawn, and text is plain ASCII for cv2.
        self.assertTrue(np.any((frame[..., 2] > 200) & (frame[..., 0] < 80)))
        self.assertEqual(plain("ep 0003 \u00b7 SUCCESS"), "ep 0003 | SUCCESS")
        untouched = np.full((240, 320, 3), 7, np.uint8)
        draw_overlay(untouched, ["", None])
        self.assertTrue((untouched == 7).all())

    def test_reuses_fresh_operator_frame_from_recorder(self):
        clock = Clock()
        view, streamer = self.attached(size=(320, 180), camera="operator", fps=30, clock=clock, budget=10.)
        recorded = np.zeros((120, 160, 3), np.uint8)
        recorded[..., 0] = 255                                  # pure red, RGB
        view.maybe_send(self.data, None, reuse=(recorded, clock.t - .01))
        self.assertEqual((view.reused, view.renderer), (1, None))   # no second render
        frame = streamer.frames[-1]
        self.assertEqual(frame.shape, (180, 320, 3))
        np.testing.assert_array_equal(frame[90, 160], [0, 0, 255])  # red in BGR
        np.testing.assert_array_equal(frame[90, 5], [0, 0, 0])      # 4:3 letterboxed in 16:9
        self.assertEqual(recorded[0, 0, 0], 255)                    # recorder frame untouched
        if not self.can_render:
            return
        clock.t += 1 / 30 + 1e-6
        view.maybe_send(self.data, None, reuse=(recorded, clock.t - .1))   # stale: render
        self.assertEqual(view.reused, 1)
        self.assertIsNotNone(view.renderer)
        # The follow camera never reuses the recorder's operator frame.
        follow, streamer2 = self.attached(size=(320, 180), fps=30, clock=clock, budget=10.)
        follow.maybe_send(self.data, self.viewer_cam(), reuse=(recorded, clock.t))
        self.assertEqual(follow.reused, 0)

    def test_headset_lines(self):
        from arm_control import WristMapper
        from view_dual_arms import AutoRecenter, headset_lines
        mapper = WristMapper()
        mapper.calibrate("left", np.eye(4), [0, 0, 0])
        lines = headset_lines({"left": True, "right": False}, mapper, self.controller)
        self.assertEqual(lines, ["L: LIVE \u00b7 FOLLOW   R: WAIT \u00b7 press C"])
        auto = AutoRecenter()
        auto.recentered("left", 10.)
        self.assertIn("FOLLOW (auto)", headset_lines({"left": True, "right": True}, mapper,
                                                     self.controller, auto_recenter=auto, now=10.5)[0])
        self.assertEqual(headset_lines({"left": True, "right": True}, mapper, self.controller,
                                       paused=True)[0][:6], "PAUSED")

        class Recorder:
            recording, last_saved = True, (3, {})
            def elapsed(self, data):
                return 72.4
            def outcome_text(self):
                return "ep 0003 \u00b7 SUCCESS"
        lines = headset_lines({"left": True, "right": True}, mapper, self.controller,
                              recorder=Recorder())
        self.assertEqual(lines[1:], ["\u25cf REC 01:12", "Last: ep 0003 \u00b7 SUCCESS"])

    def test_bad_arguments(self):
        for kwargs in ({"size": (321, 240)}, {"fps": 0}, {"fps": 120}, {"camera": "head"}):
            with self.assertRaises(ValueError):
                HeadsetView(self.model, **kwargs)

    def test_cli_requires_address(self):
        from view_dual_arms import parse_args
        for argv in (["--headset-view"], ["--address", "x", "--headset-view", "--headset-fps", "0"],
                     ["--address", "x", "--headset-view", "--headset-size", "641x480"]):
            with mock.patch.object(sys, "argv", ["view_dual_arms.py", *argv]), \
                    self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args()
        with mock.patch.object(sys, "argv", ["view_dual_arms.py", "--address", "x", "--headset-view"]):
            args = parse_args()
        self.assertEqual((args.headset_size, args.headset_fps, args.headset_camera),
                         ((960, 540), 20, "follow"))


if __name__ == "__main__":
    unittest.main()
