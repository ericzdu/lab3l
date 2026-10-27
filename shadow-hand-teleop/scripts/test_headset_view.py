"""Headset view and the bridge's streamer hook, with fake streamers (no headset, no network)."""
import contextlib
import io
import time
import unittest

import numpy as np

from hand_retargeting import synthetic_skeleton
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


if __name__ == "__main__":
    unittest.main()
