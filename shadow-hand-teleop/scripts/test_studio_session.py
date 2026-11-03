"""Studio session with a fake sim (no MuJoCo, headset or display): launch, keys, status, stop."""
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest

from studio.server import create_app
from studio.session import Session, chip, next_step

ADDRESS = "198.51.100.23"
PORT = 8765
TOKEN = "page-token"

# Speaks the control protocol through the real sim_control.ControlServer.
FAKE_SIM = textwrap.dedent('''
    import argparse, os, sys, time
    from queue import Empty, SimpleQueue
    sys.path.insert(0, sys.argv.pop(1))
    from sim_control import ControlServer
    parser = argparse.ArgumentParser()
    parser.add_argument("--address"); parser.add_argument("--control-port", type=int)
    parser.add_argument("--control-token"); parser.add_argument("--operator", default="")
    args, flags = parser.parse_known_args()
    print("Connecting to Tracking Streamer at " + args.address, flush=True)
    print("flags: " + " ".join(sorted(flags)), flush=True)
    if os.environ.get("FAKE_SIM_CRASH"):
        print("Traceback: boom", flush=True)
        sys.exit(3)
    events = SimpleQueue()
    server = ControlServer(events, args.control_port, args.control_token)
    sides = {s: {"live": True, "follow": False, "forearm": False, "reach_limit": False}
             for s in ("left", "right")}
    status = {"sides": sides, "recording": False, "rec_seconds": None, "cup_at_start": True,
              "last_saved": None, "paused": False, "error": None}
    counter = 0
    while True:
        try:
            key = events.get(timeout=.01)
        except Empty:
            key = None
        if key == 256:
            print("Esc: saving and stopping", flush=True)
            break
        if key == ord("C"):
            for side in sides.values():
                side["follow"] = True
        if key == ord("E"):
            if status["recording"]:
                status["last_saved"] = {"id": counter, "folder": "f%04d" % counter, "outcome": "success",
                                        "failure_reason": None, "confidence": .9}
                status["cup_at_start"] = False
            else:
                counter += 1
            status["recording"] = not status["recording"]
        if key == ord("H"):
            status["cup_at_start"] = True
        if key is not None:
            print("key %d" % key, flush=True)
        server.publish(json.loads(json.dumps(status)))
    server.close()
''').replace("import argparse", "import argparse, json")


def wait_for(predicate, timeout=10.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    return None


class FakeSimSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = Path(self.tmp.name)
        script = self.cwd / "fake_sim.py"
        script.write_text(FAKE_SIM)
        scripts = str(Path(__file__).resolve().parent)
        self.env = dict(os.environ)
        self.session = Session(launcher=[sys.executable, str(script), scripts], env=self.env,
                               cwd=self.cwd, stop_wait=5)
        self.addCleanup(self.session.stop)
        (self.cwd / "episodes").mkdir()
        self.app = create_app(self.cwd / "episodes", port=PORT, token=TOKEN, session=self.session)
        self.client = self.app.test_client()

    def post(self, path, body=None, token=TOKEN):
        return self.client.post(path, json=body or {}, base_url=f"http://127.0.0.1:{PORT}",
                                headers={"X-Studio-Token": token})

    def state(self):
        return self.client.get("/api/session", base_url=f"http://127.0.0.1:{PORT}").get_json()

    def press(self, key):
        self.assertTrue(wait_for(lambda: self.post("/api/session/key", {"key": key}).status_code == 200))

    def test_start_status_keys_steps_and_stop(self):
        response = self.post("/api/session/start", {"address": ADDRESS, "operator": "NT",
                                                    "auto_recenter": True, "headset_view": True,
                                                    "grasp_assist": False})
        self.assertEqual(response.status_code, 200, response.get_json())
        state = wait_for(lambda: (lambda s: s if s["status"] else None)(self.state()))
        self.assertTrue(state["running"])
        self.assertEqual(state["next_step"], "calibrate")
        self.assertEqual(state["chips"], {"left": "amber", "right": "amber"})
        self.press("C")
        state = wait_for(lambda: (lambda s: s if s["next_step"] == "record" else None)(self.state()))
        self.assertEqual(state["chips"], {"left": "green", "right": "green"})
        self.press("E")
        self.assertTrue(wait_for(lambda: self.state()["next_step"] == "recording"))
        self.press("E")
        state = wait_for(lambda: (lambda s: s if (s["status"] or {}).get("last_saved") else None)(self.state()))
        self.assertEqual(state["status"]["last_saved"]["folder"], "f0001")
        self.assertEqual(state["next_step"], "reset")          # cup moved during the attempt
        self.press("H")
        self.assertTrue(wait_for(lambda: self.state()["next_step"] == "record"))
        self.assertEqual(self.post("/api/session/key", {"key": "rm -rf"}).status_code, 400)
        # Stop sends Esc: the fake sim logs it and exits 0, not a crash.
        state = self.post("/api/session/stop").get_json()
        self.assertFalse(state["running"])
        self.assertEqual((state["exit_code"], state["crashed"]), (0, False))
        log = self.client.get("/api/session/log", base_url=f"http://127.0.0.1:{PORT}").get_json()["lines"]
        self.assertIn("Esc: saving and stopping", log)
        self.assertIn("Connecting to Tracking Streamer at <address>", log)
        # Flags Studio passes to the viewer.
        flags = next(line for line in log if line.startswith("flags: ")).split()[1:]
        self.assertEqual(sorted(flags), sorted(["--wrist", "human", "--workspace", "--clean-ui", "--no-panel",
                                                "--auto-recenter", "--headset-view"]))
        argv = self.session.process.args
        self.assertEqual(argv[argv.index("--address") + 1], ADDRESS)    # one argument, no shell
        self.assertNotIn("--grasp-assist", argv)
        # The address appears nowhere in logs, state or files.
        self.assertNotIn(ADDRESS, json.dumps([log, state]))
        for path in self.cwd.rglob("*"):
            if path.is_file():
                self.assertNotIn(ADDRESS, path.read_text(errors="ignore"), path)
        self.assertIsNone(self.session._address)

    def test_only_one_session_and_token_required(self):
        self.assertEqual(self.post("/api/session/start", {"address": ADDRESS}, token="bad").status_code, 403)
        self.assertEqual(self.post("/api/session/start", {"address": ""}).status_code, 400)
        self.assertEqual(self.post("/api/session/start", {"address": "--seconds"}).status_code, 400)
        self.assertEqual(self.post("/api/session/start", {"address": ADDRESS, "grasp_assist": True}).status_code, 200)
        self.assertIn("--grasp-assist", self.session.process.args)
        second = self.post("/api/session/start", {"address": ADDRESS})
        self.assertEqual(second.status_code, 409)
        self.assertIn("already running", second.get_json()["error"])

    def test_crash_shows_exit_code(self):
        self.env["FAKE_SIM_CRASH"] = "1"
        self.assertEqual(self.post("/api/session/start", {"address": ADDRESS}).status_code, 200)
        state = wait_for(lambda: (lambda s: s if s["exit_code"] is not None else None)(self.state()))
        self.assertFalse(state["running"])
        self.assertEqual((state["exit_code"], state["crashed"]), (3, True))
        log = wait_for(lambda: (lambda lines: lines if any("exited with code 3" in l for l in lines) else None)(
            self.session.log_lines()))
        self.assertIn("Traceback: boom", log)
        self.assertNotIn(ADDRESS, "\n".join(log))
        # A new session can start after a crash.
        del self.env["FAKE_SIM_CRASH"]
        self.assertEqual(self.post("/api/session/start", {"address": ADDRESS}).status_code, 200)

    def test_page_gets_the_token(self):
        html = self.client.get("/", base_url=f"http://127.0.0.1:{PORT}").get_data(as_text=True)
        self.assertIn(f'<meta name="studio-token" content="{TOKEN}">', html)
        self.assertNotIn("{{STUDIO_TOKEN}}", html)
        for name in ("app.js", "api.js", "session.js", "result.js", "timeline.js", "episodes.js", "app.css"):
            with self.client.get(f"/static/{name}", base_url=f"http://127.0.0.1:{PORT}") as response:
                self.assertEqual(response.status_code, 200, name)


class StepGuideTests(unittest.TestCase):
    def status(self, follow=(True, True), live=(True, True), cup_at_start=True, recording=False):
        return {"sides": {side: {"live": l, "follow": f} for side, l, f in zip(("left", "right"), live, follow)},
                "cup_at_start": cup_at_start, "recording": recording}

    def test_next_step(self):
        self.assertIsNone(next_step(None))
        self.assertEqual(next_step(self.status()), "record")
        self.assertEqual(next_step(self.status(follow=(True, False))), "calibrate")
        self.assertEqual(next_step(self.status(cup_at_start=False, follow=(False, False))), "reset")
        self.assertEqual(next_step(self.status(cup_at_start=False, recording=True)), "recording")
        self.assertEqual(next_step({"cup_at_start": True}), "calibrate")    # no side info yet

    def test_chips(self):
        self.assertEqual(chip({"live": True, "follow": True}), "green")
        self.assertEqual(chip({"live": True, "follow": False}), "amber")
        self.assertEqual(chip({"live": False, "follow": True}), "grey")
        self.assertEqual(chip(None), "grey")


if __name__ == "__main__":
    unittest.main()
