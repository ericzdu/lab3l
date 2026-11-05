"""lab3l Studio episode API (demo cut) with the Flask test client and synthetic episodes."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from studio.episodes import FRIENDLY_REASONS, phase_strip
from studio.server import create_app

PORT = 8765
TOKEN = "test-token"
SEGMENTS = [("idle", 0, 1), ("reach", 1, 2.1), ("grasp", 2.1, 3.2), ("lift", 3.2, 3.6),
            ("transport", 3.6, 7.4), ("place", 7.4, 9.3), ("release", 9.3, 9.6), ("retreat", 9.6, 10)]


def make_episode(root, number, outcome=None, reason=None, placement=None, minute=0, video=True):
    folder = Path(root) / f"20261102-12{minute:02d}00_cup_to_tray_{number:04d}"
    folder.mkdir(parents=True)
    labels = None
    if outcome is not None:
        labels = {"labeler_version": "2",
                  "outcome": {"success": outcome, "failure_reason": reason, "confidence": .8,
                              "placement": placement},
                  "segments": [{"phase": p, "side": "right", "start_t": a, "end_t": b, "confidence": .9}
                               for p, a, b in SEGMENTS],
                  "events": [{"t": 2.1, "kind": "first_touch", "side": "right"},
                             {"t": 3.2, "kind": "lift_off", "side": "right"}]}
    metadata = {"episode_id": folder.name, "created_utc": f"2026-11-02T12:{minute:02d}:00Z",
                "duration_sim_s": 10.0, "source": "live", "grasp_assist": False,
                "video": {"fps": 30.0}, "labels": labels}
    (folder / "episode.json").write_text(json.dumps(metadata))
    with h5py.File(folder / "episode.h5", "w") as file:
        file["time/sim"] = np.linspace(0, 10, 501)
        file["video/overview/frame_time"] = np.arange(300) / 30 + .05
        file["video/operator/frame_time"] = np.arange(300) / 30 + .05
    if video:
        for camera in ("overview", "operator"):
            (folder / f"cam_{camera}.mp4").write_bytes(bytes(range(256)) * 40)
    return folder


class StudioApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "episodes"
        self.success = make_episode(self.root, 16, True, placement="placed", minute=1)
        self.fail = make_episode(self.root, 17, False, "dropped_outside_tray", minute=2)
        self.unlabeled = make_episode(self.root, 18, minute=3, video=False)
        make_episode(self.root / "_junk", 15, True, minute=0)
        outside = make_episode(Path(self.tmp.name) / "elsewhere", 99, True, minute=9)
        os.symlink(outside, self.root / "20261102-125900_cup_to_tray_0099")
        self.app = create_app(self.root, port=PORT, token=TOKEN)
        self.client = self.app.test_client()

    def get(self, path, **kwargs):
        return self.client.get(path, base_url=f"http://127.0.0.1:{PORT}", **kwargs)

    def test_list_counts_and_order(self):
        body = self.get("/api/episodes").get_json()
        self.assertEqual([row["number"] for row in body["episodes"]], [18, 17, 16])
        self.assertEqual((body["count"], body["successes"]), (3, 1))
        rows = {row["number"]: row for row in body["episodes"]}
        self.assertEqual(rows[17]["friendly_reason"], "dropped outside tray")
        self.assertEqual(rows[16]["placement"], "placed")
        self.assertEqual(rows[18]["outcome"], "unlabeled")

    def test_detail_has_segments_events_frame_time(self):
        body = self.get(f"/api/episodes/{self.fail.name}").get_json()
        self.assertEqual(body["outcome"], "fail")
        self.assertEqual(body["friendly_reason"], "dropped outside tray")
        self.assertEqual([s["phase"] for s in body["segments"]], [p for p, _, _ in SEGMENTS])
        self.assertEqual([e["kind"] for e in body["events"]], ["first_touch", "lift_off"])
        self.assertEqual(len(body["cameras"]["overview"]["frame_time"]), 300)
        self.assertAlmostEqual(body["cameras"]["overview"]["frame_time"][1], .05 + 1 / 30, places=4)
        self.assertEqual(body["video_fps"], 30.0)
        self.assertEqual(body["phase_strip"],
                         "reach 1.1 s · grasp 1.1 s · lift · transport 3.8 s · place 1.9 s · release · retreat")
        self.assertEqual(self.get(f"/api/episodes/{self.unlabeled.name}").get_json()["cameras"], {})

    def test_friendly_reasons_cover_the_labeler(self):
        from episode_labels import FAILURE_ORDER
        self.assertEqual(set(FRIENDLY_REASONS), set(FAILURE_ORDER))
        self.assertEqual(phase_strip([]), "")

    def test_mp4_range_returns_206(self):
        response = self.get(f"/api/episodes/{self.success.name}/video/overview",
                            headers={"Range": "bytes=100-199"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.headers["Content-Range"], "bytes 100-199/10240")
        self.assertEqual(response.data, (bytes(range(256)) * 40)[100:200])
        self.assertEqual(response.mimetype, "video/mp4")
        response.close()
        self.assertEqual(self.get(f"/api/episodes/{self.unlabeled.name}/video/overview").status_code, 404)

    def test_host_and_token_checks(self):
        for host in ("evil.example:8765", "127.0.0.1:9999", "localhost"):
            self.assertEqual(self.client.get("/api/episodes", base_url=f"http://{host}").status_code, 403)
        self.assertEqual(self.client.get("/api/episodes", base_url=f"http://localhost:{PORT}").status_code, 200)
        url = f"http://127.0.0.1:{PORT}"
        self.assertEqual(self.client.post("/api/episodes", base_url=url).status_code, 403)
        self.assertEqual(self.client.post("/api/episodes", base_url=url,
                                          headers={"X-Studio-Token": "nope"}).status_code, 403)
        self.assertEqual(self.client.post("/api/episodes", base_url=url,
                                          headers={"X-Studio-Token": TOKEN}).status_code, 405)

    def test_path_traversal_and_escapes_rejected(self):
        for folder in ("..", "..%2F..%2Fetc", "%2Fetc%2Fpasswd", "_junk",
                       "20261102-120000_cup_to_tray_0015",          # lives in _junk
                       "20261102-125900_cup_to_tray_0099"):         # symlink outside the root
            self.assertEqual(self.get(f"/api/episodes/{folder}").status_code, 404, folder)
        self.assertEqual(self.get(f"/api/episodes/{self.success.name}/video/..%2Fepisode.h5").status_code, 404)
        self.assertEqual(self.get(f"/api/episodes/{self.success.name}/video/top").status_code, 404)
        self.assertEqual(self.get("/static/..%2F..%2Fserver.py").status_code, 404)


if __name__ == "__main__":
    unittest.main()
