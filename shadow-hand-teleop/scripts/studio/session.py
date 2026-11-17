"""One live sim session for lab3l Studio: start/stop the mjpython viewer, relay keys and status.

The sim runs as `mjpython scripts/view_dual_arms.py --wrist human --workspace
--clean-ui --no-panel --address ... --control-port P --control-token T` (an
argument list, never a shell). Studio talks to it over the localhost control
channel (sim_control.py). The tracking address lives only in this process's
memory and the sim's argument list: log lines are scrubbed of it and nothing
is written to disk.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
LOG_LINES = 200
STOP_WAIT = 15.     # s for the sim to save an open episode after Esc
KEYS = ("C", "E", "X", "H", "B", "space")


def default_launcher():
    return [str(PROJECT / ".venv" / "bin" / "mjpython"), str(PROJECT / "scripts" / "view_dual_arms.py")]


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def next_step(status):
    """Step guide: 'reset' | 'calibrate' | 'record' | 'recording', or None without a session.

    Recording wins (the cup moves during an attempt); otherwise a cup away
    from its start needs Reset, then any arm without FOLLOW needs Calibrate,
    else the operator can Record.
    """
    if not status:
        return None
    if status.get("recording"):
        return "recording"
    if status.get("cup_at_start") is False:
        return "reset"
    sides = status.get("sides") or {}
    if not sides or not all(side.get("follow") for side in sides.values()):
        return "calibrate"
    return "record"


def chip(side_status):
    """Hand chip colour: green LIVE + FOLLOW, amber LIVE but needs C, grey no tracking."""
    if not side_status or not side_status.get("live"):
        return "grey"
    return "green" if side_status.get("follow") else "amber"


class Session:
    def __init__(self, launcher=None, env=None, cwd=PROJECT, stop_wait=STOP_WAIT):
        self.launcher = list(launcher or default_launcher())
        self.env, self.cwd, self.stop_wait = env, cwd, stop_wait
        self.lock = threading.Lock()
        self.process = None
        self.log = deque(maxlen=LOG_LINES)
        self.status = None
        self.exit_code = None
        self.crashed = False
        self.stopping = False
        self.options = {}
        self._address = None
        self._token = None
        self._socket = None

    # -- lifecycle --------------------------------------------------------
    def running(self):
        return self.process is not None and self.process.poll() is None

    def command(self, address, port, token, operator="", auto_recenter=True, headset_view=True,
                grasp_assist=False):
        argv = [*self.launcher, "--wrist", "human", "--workspace", "--clean-ui", "--no-panel",
                "--address", address, "--control-port", str(port), "--control-token", token]
        if operator:
            argv += ["--operator", operator]
        if auto_recenter:
            argv.append("--auto-recenter")
        if headset_view:
            argv.append("--headset-view")
        if grasp_assist:
            argv.append("--grasp-assist")
        return argv

    def start(self, address, operator="", auto_recenter=True, headset_view=True, grasp_assist=False):
        address = str(address or "").strip()
        operator = str(operator or "").strip()
        if not address or len(address) > 200 or any(c.isspace() for c in address) or address.startswith("-"):
            raise ValueError("Enter the Tracking Streamer IP or room code")
        if len(operator) > 32 or operator.startswith("-"):
            raise ValueError("Operator tag must be at most 32 characters")
        with self.lock:
            if self.running():
                raise RuntimeError("A session is already running")
            port, token = free_port(), secrets.token_urlsafe(24)
            env = dict(os.environ if self.env is None else self.env)
            env.setdefault("MENAGERIE_CACHE_DIR", str(PROJECT / ".cache" / "mujoco_menagerie"))
            env.update(GRPC_VERBOSITY="ERROR", GLOG_minloglevel="2", PYTHONUNBUFFERED="1")
            self.log.clear()
            self.status, self.exit_code, self.crashed, self.stopping = None, None, False, False
            self.options = {"operator": operator, "auto_recenter": bool(auto_recenter),
                            "headset_view": bool(headset_view), "grasp_assist": bool(grasp_assist)}
            self._address, self._token = address, token
            self.process = subprocess.Popen(
                self.command(address, port, token, operator, auto_recenter, headset_view, grasp_assist),
                cwd=self.cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                # Own process group: a terminal Ctrl-C/hang-up reaches Studio, which
                # then stops the sim with Esc so an open episode is saved.
                start_new_session=True)
            process = self.process
        threading.Thread(target=self._read_log, args=(process,), daemon=True).start()
        threading.Thread(target=self._control, args=(process, port), daemon=True).start()

    def stop(self):
        """Esc first (the sim saves any open episode), then SIGINT, then kill."""
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return self.exit_code
            self.stopping = True
        if not self.send_key("esc"):
            process.send_signal(2)                   # no channel yet: Ctrl-C also saves
        for signal_step in (None, 2, 9):
            if signal_step is not None and process.poll() is None:
                process.send_signal(signal_step)
            try:
                process.wait(timeout=self.stop_wait if signal_step is None else 5)
                break
            except subprocess.TimeoutExpired:
                continue
        return process.returncode

    # -- keys and status ---------------------------------------------------
    def send_key(self, key):
        if key not in (*KEYS, "esc"):
            raise ValueError(f"unknown key {key!r}")
        sock = self._socket
        if sock is None or not self.running():
            return False
        try:
            sock.sendall((json.dumps({"token": self._token, "key": key}) + "\n").encode())
            return True
        except OSError:
            return False

    def _control(self, process, port):
        sock = None
        while process.poll() is None and sock is None:
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=1)
            except OSError:
                time.sleep(.2)
        if sock is None:
            return
        self._socket = sock
        sock.settimeout(None)
        try:
            with sock.makefile("r") as stream:
                for line in stream:
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(message.get("status"), dict):
                        self.status = message["status"]
        except OSError:
            pass
        finally:
            if self._socket is sock:
                self._socket = None
            sock.close()

    def scrub(self, line):
        from episode_recorder import scrub
        line = scrub(line.rstrip("\n"))
        for secret in (self._address, self._token):
            if secret:
                line = line.replace(secret, "<address>" if secret == self._address else "<token>")
        return line

    def _read_log(self, process):
        with process.stdout:
            for line in process.stdout:
                self.log.append(self.scrub(line))
        code = process.wait()
        with self.lock:
            if process is self.process:
                self.exit_code = code
                self.crashed = code != 0 and not self.stopping
                self._address = None             # every log line is scrubbed; forget it
        self.log.append(f"[sim exited with code {code}]")

    def state(self):
        """JSON for the page: running, status, step guide, chips, exit code."""
        running = self.running()
        status = self.status if running else None
        sides = (status or {}).get("sides") or {}
        return {"running": running, "stopping": self.stopping and running, "status": status,
                "next_step": next_step(status) if running else None,
                "chips": {side: chip(sides.get(side)) for side in ("left", "right")},
                "exit_code": self.exit_code, "crashed": self.crashed, "options": self.options}

    def log_lines(self):
        return list(self.log)
