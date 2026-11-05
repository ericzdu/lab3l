"""Exercise the real control/status pipes without opening native GUI windows."""

import subprocess
import sys
from queue import SimpleQueue
import unittest
from unittest.mock import patch

from control_panel import KEYS, LIVE_KEYS, ControlPanel, button_grid


class PanelTests(unittest.TestCase):
    def test_commands_status_and_graceful_shutdown(self):
        child = '''
import json, sys
print('not a command', flush=True)
for value in (90, '73', 49, 73):
    print(json.dumps({'key': value}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message.get('close'):
        break
    if message.get('status') == 'Ready':
        print(json.dumps({'key': 66}), flush=True)
'''
        original = subprocess.Popen

        def factory(command, **kwargs):
            self.assertIn('--gui', command)
            self.assertIn('--live', command)
            return original([sys.executable, '-u', '-c', child], **kwargs)

        events = SimpleQueue()
        with patch('control_panel.subprocess.Popen', side_effect=factory):
            panel = ControlPanel(events, live=True)
        try:
            self.assertEqual(events.get(timeout=3), 49)
            self.assertEqual(events.get(timeout=3), 73)
            panel.status('Ready')
            self.assertEqual(events.get(timeout=3), 66)
        finally:
            panel.close()
        self.assertEqual(panel.process.returncode, 0)
        self.assertTrue(events.empty())

    def test_record_keys_and_live_allow_list(self):
        for key in "EX":
            self.assertIn(ord(key), KEYS)
            self.assertIn(ord(key), LIVE_KEYS)
        self.assertEqual(LIVE_KEYS, set(map(ord, "12HCBEX ")))
        self.assertTrue(LIVE_KEYS <= KEYS)
        # E/X must stay free of the robot movement keys.
        self.assertFalse(set(map(ord, "EX")) & set(map(ord, "WSADRVIKJLUPFO")))

    def test_recording_flag_enables_record_buttons(self):
        original = subprocess.Popen
        commands = []

        def factory(command, **kwargs):
            commands.append(command)
            return original([sys.executable, '-c', 'import sys; sys.stdin.read()'], **kwargs)

        for recording in (True, False):
            with patch('control_panel.subprocess.Popen', side_effect=factory):
                panel = ControlPanel(SimpleQueue(), recording=recording)
            panel.close()
        self.assertIn('--recording', commands[0])
        self.assertNotIn('--recording', commands[1])
        for live in (True, False):
            enabled = {key: on for _, key, on in button_grid(live, recording=True)}
            self.assertTrue(enabled[ord("E")] and enabled[ord("X")])
            enabled = {key: on for _, key, on in button_grid(live, recording=False)}
            self.assertFalse(enabled[ord("E")] or enabled[ord("X")])

    def test_compact_live_layout(self):
        live = button_grid(live=True, recording=True)
        self.assertEqual([key for _, key, _ in live], [ord(k) for k in "CEXH"] + [32, ord("B")])
        self.assertTrue(all(enabled for _, _, enabled in live))
        self.assertEqual([text for text, _, _ in live],
                         ["Calibrate · C", "Record / Stop · E", "Discard · X", "Reset · H",
                          "Pause · Space", "Switch view · B"])
        # Every live button is a key the live keyboard filter accepts.
        self.assertTrue({key for _, key, _ in live} <= LIVE_KEYS)

    def test_keyboard_layout_unchanged(self):
        keyboard = button_grid(live=False, recording=True)
        self.assertEqual(len(keyboard), 20)
        self.assertEqual([key for _, key, _ in keyboard[:14]], [ord(k) for k in "WSADRVIKJLUPFO"])
        enabled = {key: on for _, key, on in keyboard}
        self.assertFalse(enabled[ord("C")])            # calibrate is live-only
        self.assertTrue(all(enabled[ord(k)] for k in "WSADRVIKJLUPFOHBEX"))

    def test_unexpected_panel_exit_requests_stop(self):
        original = subprocess.Popen
        events = SimpleQueue()
        with patch('control_panel.subprocess.Popen', side_effect=lambda command, **kwargs:
                   original([sys.executable, '-c', 'pass'], **kwargs)):
            panel = ControlPanel(events)
        try:
            self.assertEqual(events.get(timeout=3), 256)
        finally:
            panel.close()


if __name__ == '__main__':
    unittest.main()
