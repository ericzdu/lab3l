"""Native controls in a separate process, isolated from MuJoCo hotkeys.

Tk runs in a normal Python main thread/process on macOS, while mjpython owns
MuJoCo's native GUI loop. Small JSON messages carry commands and status only.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading


KEYS = set(map(ord, "12WASDRVIKJLUPFOHCB ")) | {256}


class ControlPanel:
    def __init__(self, events, live=False):
        self.events = events
        self.closed = False
        python = Path(sys.prefix) / "bin" / "python"
        command = [str(python), str(Path(__file__).resolve()), "--gui"]
        if live:
            command.append("--live")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                    key = message.get("key")
                    if type(key) is int and key in KEYS:
                        self.events.put(key)
                except (ValueError, AttributeError):
                    continue
        finally:
            if not self.closed:
                print("Hand controls window closed; stopping simulation.", flush=True)
                self.events.put(256)  # GLFW Escape; no native viewer key event.

    def _send(self, message):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def status(self, text):
        self._send({"status": text})

    def close(self):
        self.closed = True
        self._send({"close": True})
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=2)
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()


def gui(live):
    import tkinter as tk
    from tkinter import ttk
    from queue import Empty, SimpleQueue

    root = tk.Tk()
    root.title("Holos — Hand controls")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=14)
    frame.grid(sticky="nsew")
    ttk.Label(frame, text="Hand controls", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(frame, text="Click this window before using the keyboard.\nUse the simulation window for mouse/camera controls.").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))
    selected = tk.StringVar(value="right")
    status = tk.StringVar(value="Waiting for hand tracking…" if live else "Offline keyboard mode")

    def emit(key):
        print(json.dumps({"key": key}), flush=True)
        if key in (ord("1"), ord("2")):
            selected.set("left" if key == ord("1") else "right")
        # Keep Space from also activating whichever button was last clicked.
        root.focus_set()

    for col, side in enumerate(("left", "right")):
        ttk.Radiobutton(frame, text=f"{col + 1}: {side.title()} arm", variable=selected, value=side,
            takefocus=False, command=lambda key=ord("1") + col: emit(key)).grid(row=2, column=col, sticky="w", pady=(0, 8))

    rows = [
        ("Forward · W", "W", "Back · S", "S"),
        ("Left · A", "A", "Right · D", "D"),
        ("Up · R", "R", "Down · V", "V"),
        ("Bend up · I", "I", "Bend down · K", "K"),
        ("Tilt left · J", "J", "Tilt right · L", "L"),
        ("Turn palm · U", "U", "Turn back · P", "P"),
        ("Close slowly · F", "F", "Open slowly · O", "O"),
    ]
    for row, items in enumerate(rows, start=3):
        for col in range(2):
            text, key = items[2 * col: 2 * col + 2]
            ttk.Button(frame, text=text, width=20, takefocus=False, state="disabled" if live else "normal",
                command=lambda key=key: emit(ord(key))).grid(row=row, column=col, padx=3, pady=3, sticky="ew")
    controls = [
        ("Home · H", ord("H")), ("Calibrate · C", ord("C")),
        ("Pause / resume · Space", 32), ("Switch view · B", ord("B")),
    ]
    for i, (text, key) in enumerate(controls):
        ttk.Button(frame, text=text, width=20, takefocus=False,
            state="disabled" if key == ord("C") and not live else "normal",
            command=lambda key=key: emit(key)).grid(row=10 + i // 2, column=i % 2, padx=3, pady=3, sticky="ew")
    ttk.Label(frame, textvariable=status, wraplength=340).grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 4))
    if live:
        ttk.Label(frame, text="Press C here to calibrate.\nForearm ON = usable data + calibrated guide;\nunavailable = wrist-only fallback.",
            wraplength=340).grid(row=13, column=0, columnspan=2, sticky="w")

    def keypress(event):
        # Keep ordinary copy/menu shortcuts from becoming robot commands.
        if event.state & (0x4 | 0x8 | 0x10 | 0x80 | 0x20000):
            return
        if event.keysym == "Escape":
            shutdown()
            return "break"
        if event.char:
            text = event.char.upper()
            if len(text) == 1 and ord(text) in KEYS:
                key = ord(text)
                if not live or key in set(map(ord, "12HCB ")):
                    emit(key)
                return "break"

    def shutdown():
        emit(256)
        root.destroy()

    root.bind_all("<KeyPress>", keypress)
    root.protocol("WM_DELETE_WINDOW", shutdown)
    incoming = SimpleQueue()

    def read_parent():
        for line in sys.stdin:
            try:
                incoming.put(json.loads(line))
            except ValueError:
                pass
        incoming.put({"close": True})

    def poll():
        while True:
            try:
                message = incoming.get_nowait()
            except Empty:
                break
            if message.get("close"):
                root.destroy()
                return
            if "status" in message:
                status.set(str(message["status"]))
        root.after(50, poll)

    threading.Thread(target=read_parent, daemon=True).start()
    root.after(50, poll)
    root.after_idle(root.focus_set)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.gui:
        gui(args.live)
