"""Exercise the real control/status pipes without opening native GUI windows."""

import subprocess
import sys
from queue import SimpleQueue
import unittest
from unittest.mock import patch

from control_panel import KEYS, LIVE_KEYS, ControlPanel


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
        from inspect import getsource
        import control_panel
        self.assertIn('("Record · E", ord("E"))', getsource(control_panel.gui))
        self.assertIn('("Discard · X", ord("X"))', getsource(control_panel.gui))

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
