"""ControlServer over a real localhost socket; no simulation or display."""
import contextlib
import io
import json
from queue import Empty, SimpleQueue
import socket
import sys
import time
import unittest
from unittest import mock

from sim_control import KEY_CODES, ControlServer


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


def wait_for(predicate, timeout=3.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.01)
    return False


class ControlServerTests(unittest.TestCase):
    def setUp(self):
        self.events = SimpleQueue()
        self.clock = Clock()
        self.server = ControlServer(self.events, 0, "secret", clock=self.clock)
        self.addCleanup(self.server.close)

    def connect(self):
        client = socket.create_connection(("127.0.0.1", self.server.port), timeout=3)
        self.addCleanup(client.close)
        self.assertTrue(wait_for(lambda: self.server.client is not None))
        return client

    def send(self, client, **message):
        client.sendall((json.dumps(message) + "\n").encode())

    def test_keys_need_token_and_map_to_glfw_codes(self):
        client = self.connect()
        self.send(client, token="wrong", key="C")
        self.send(client, key="C")
        self.send(client, token="secret", key="Q")          # unknown key
        client.sendall(b"not json\n")
        for key in ("C", "E", "X", "H", "B", "space", "1", "2", "esc"):
            self.send(client, token="secret", key=key)
        received = [self.events.get(timeout=3) for _ in range(9)]
        self.assertEqual(received, [67, 69, 88, 72, 66, 32, 49, 50, 256])
        self.assertEqual(received, list(KEY_CODES.values()))
        with self.assertRaises(Empty):
            self.events.get(timeout=.2)

    def read_statuses(self, client, count, timeout=3.):
        client.settimeout(timeout)
        stream = client.makefile("r")
        return [json.loads(stream.readline())["status"] for _ in range(count)]

    def test_status_on_change_at_most_10_hz(self):
        client = self.connect()
        self.assertTrue(self.server.publish({"n": 1}))
        self.assertFalse(self.server.publish({"n": 1}))     # unchanged
        self.assertTrue(wait_for(self.server.outbox.empty))  # sent; an unsent one would be replaced
        self.clock.now = .05
        self.assertFalse(self.server.publish({"n": 2}))     # too soon
        self.clock.now = .1
        self.assertTrue(self.server.publish({"n": 2}))
        statuses = self.read_statuses(client, 2)
        self.assertEqual(statuses, [{"n": 1}, {"n": 2}])
        sent = 0
        for i in range(100):                                 # 1 s of 100 Hz changes
            self.clock.now = 1 + i * .01
            sent += self.server.publish({"n": 10 + i})
        self.assertLessEqual(sent, 10)

    def test_no_client_and_slow_client_never_block(self):
        self.assertFalse(self.server.publish({"n": 0}))      # nobody connected
        client = self.connect()
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        start = time.perf_counter()
        for i in range(2000):                                # client never reads
            self.clock.now = i
            self.server.publish({"n": i, "pad": "x" * 2000})
        self.assertLess(time.perf_counter() - start, .5)

    def test_disconnect_then_new_client(self):
        first = self.connect()
        first.close()
        self.assertTrue(wait_for(lambda: self.server.client is None))
        self.clock.now = 5
        self.assertFalse(self.server.publish({"n": 1}))
        second = self.connect()
        self.server.publish({"n": 1})
        self.assertEqual(self.read_statuses(second, 1), [{"n": 1}])
        self.send(second, token="secret", key="H")
        self.assertEqual(self.events.get(timeout=3), 72)

    def test_second_client_is_refused_while_one_is_connected(self):
        self.connect()
        other = socket.create_connection(("127.0.0.1", self.server.port), timeout=3)
        self.addCleanup(other.close)
        other.settimeout(3)
        self.assertEqual(other.recv(10), b"")                # closed by the server
        self.assertEqual(self.server.accepted, 1)

    def test_refuses_non_loopback_bind_and_empty_token(self):
        for host in ("0.0.0.0", "localhost", "::"):
            with self.assertRaises(ValueError):
                ControlServer(SimpleQueue(), 0, "t", host=host)
        with self.assertRaises(ValueError):
            ControlServer(SimpleQueue(), 0, "")


class ViewerFlagsTests(unittest.TestCase):
    def test_control_flags(self):
        from view_dual_arms import parse_args
        with mock.patch.object(sys, "argv", ["view_dual_arms.py"]):
            self.assertIsNone(parse_args().control_port)
        for argv in (["--control-port", "9000"], ["--control-port", "0", "--control-token", "t"]):
            with mock.patch.object(sys, "argv", ["view_dual_arms.py", *argv]), \
                    self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args()

    def test_control_status_has_no_address_and_reports_last_saved(self):
        from arm_control import SIDES, WristMapper
        from view_dual_arms import control_status, setup
        model, data, controller, _ = setup("human", workspace=True)
        import mujoco
        mujoco.mj_forward(model, data)

        class Recorder:
            recording, counter, notice = False, 7, None
            last_saved = (7, {"episode_id": "20261102-120000_cup_to_tray_0007",
                              "labels": {"outcome": {"success": False, "failure_reason": "never_lifted",
                                                     "confidence": .8}}})

            def elapsed(self, data):
                return 0.

        status = control_status(data, {side: True for side in SIDES}, WristMapper(), controller,
                                error="Cannot reach 10.0.0.5", recorder=Recorder(), workspace=True,
                                address="10.0.0.5")
        self.assertNotIn("10.0.0.5", json.dumps(status))
        self.assertTrue(status["cup_at_start"])
        self.assertEqual(status["sides"]["left"], {"live": True, "follow": False, "forearm": False,
                                                   "reach_limit": False})
        self.assertEqual(status["last_saved"], {"id": 7, "folder": "20261102-120000_cup_to_tray_0007",
                                                "outcome": "fail", "failure_reason": "never_lifted",
                                                "confidence": .8})
        data.qpos[model.joint("cup_free").qposadr[0]] += .1
        mujoco.mj_forward(model, data)
        self.assertFalse(control_status(data, {side: False for side in SIDES}, WristMapper(),
                                        controller, workspace=True)["cup_at_start"])


if __name__ == "__main__":
    unittest.main()
