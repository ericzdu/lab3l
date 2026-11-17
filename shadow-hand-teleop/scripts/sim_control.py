"""Localhost control channel for view_dual_arms (--control-port), used by lab3l Studio.

One client at a time on 127.0.0.1, JSON lines:

- client -> sim: {"token": T, "key": "C"} with keys C E X H B space 1 2 esc.
  Each is put on the viewer's existing `events` queue as its GLFW code, so it
  behaves exactly like the Hand controls buttons. Messages with a wrong token
  or an unknown key are ignored.
- sim -> client: {"status": {...}} on change, at most 10 per second.

The control loop only calls publish(), which never blocks: it keeps the newest
status, and a writer thread sends it (an old unsent status is dropped). If the
client is slow or disconnects, the simulation carries on.
"""
from __future__ import annotations

import hmac
import json
import queue
import socket
import threading
import time

# Same keys as the Hand controls window, as GLFW key codes.
KEY_CODES = {"C": ord("C"), "E": ord("E"), "X": ord("X"), "H": ord("H"), "B": ord("B"),
             "space": 32, "1": ord("1"), "2": ord("2"), "esc": 256}
BIND_HOST = "127.0.0.1"
MAX_LINE = 4096


class ControlServer:
    def __init__(self, events, port, token, host=BIND_HOST, min_interval=.1, clock=time.monotonic):
        if host != BIND_HOST:
            raise ValueError(f"control channel binds {BIND_HOST} only")
        if not token:
            raise ValueError("control channel needs a token")
        self.events, self.token = events, str(token)
        self.min_interval, self.clock = min_interval, clock
        self.listener = socket.create_server((host, int(port)))
        self.port = self.listener.getsockname()[1]
        self.client = None
        self.outbox = queue.Queue(maxsize=1)
        self.last_status = None
        self.last_sent = -float("inf")
        self.stopping = threading.Event()
        self.accepted = 0
        self.thread = threading.Thread(target=self._serve, name="control-server", daemon=True)
        self.thread.start()

    # -- control loop side -------------------------------------------------
    def publish(self, status):
        """Offer the newest status; returns True if it was queued. Never blocks."""
        if self.client is None or status == self.last_status:
            return False
        now = self.clock()
        if now - self.last_sent < self.min_interval:
            return False
        message = json.dumps({"status": status}, default=str) + "\n"
        outbox = self.outbox
        try:
            outbox.get_nowait()               # drop an unsent older status
        except queue.Empty:
            pass
        try:
            outbox.put_nowait(message)
        except queue.Full:
            return False
        self.last_status, self.last_sent = status, now
        return True

    # -- threads ----------------------------------------------------------
    def _serve(self):
        self.listener.settimeout(.2)
        while not self.stopping.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if self.client is not None:          # one client at a time
                client.close()
                continue
            client.settimeout(1.)
            outbox = queue.Queue(maxsize=1)      # per client: a stale session cannot eat its messages
            self.last_status = None              # a new client gets the current status
            self.outbox, self.client = outbox, client
            self.accepted += 1
            threading.Thread(target=self._session, args=(client, outbox), name="control-client",
                             daemon=True).start()

    def _session(self, client, outbox):
        writer = threading.Thread(target=self._write, args=(client, outbox), name="control-writer",
                                  daemon=True)
        writer.start()
        self._read(client)
        if self.client is client:
            self.client = None
        try:
            client.close()
        except OSError:
            pass
        writer.join(timeout=2)

    def _read(self, client):
        buffer = b""
        while not self.stopping.is_set():
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self.handle(line)
            if len(buffer) > MAX_LINE:
                buffer = b""

    def handle(self, line):
        """Validate one JSON line and queue its key; anything else is ignored."""
        try:
            message = json.loads(line)
            token, key = message.get("token"), message.get("key")
        except (ValueError, AttributeError):
            return False
        if not isinstance(token, str) or not hmac.compare_digest(token.encode(), self.token.encode()):
            return False
        code = KEY_CODES.get(key)
        if code is None:
            return False
        self.events.put(code)
        return True

    def _write(self, client, outbox):
        while not self.stopping.is_set() and self.client is client:
            try:
                message = outbox.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                client.sendall(message.encode())
            except (socket.timeout, OSError):
                # Too slow (1 s send timeout) or gone: drop this client; the
                # reader sees the shutdown and the server accepts a new one.
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    def close(self):
        self.stopping.set()
        for sock in (self.client, self.listener):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        self.thread.join(timeout=2)
