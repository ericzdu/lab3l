"""Table of recorded episodes for picking good/bad demonstrations.

Reads each episode.json sidecar only. Labels are heuristic; re-run
label_episode.py after changing the labeler. Episodes are simulated robot
state driven by human tracking, not real-robot data.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

from episode_labels import PHASES
from episode_recorder import DEFAULT_RECORD_DIR


COLUMNS = (["id", "operator", "source", "duration_s", "success", "failure_reason", "confidence"]
           + [f"t_{phase}" for phase in PHASES] + ["regrasps", "assist", "dropped_frames"])


def episode_row(metadata):
    labels = metadata.get("labels") or {}
    outcome = labels.get("outcome") or {}
    phase_time = {phase: 0. for phase in PHASES}
    for segment in labels.get("segments") or []:
        phase_time[segment["phase"]] += segment["end_t"] - segment["start_t"]
    dropped = (metadata.get("video") or {}).get("dropped_frames")
    confidence = outcome.get("confidence")
    return {
        "id": metadata.get("episode_id"), "operator": metadata.get("operator", ""),
        "source": metadata.get("source"), "duration_s": round(metadata.get("duration_sim_s", 0.), 2),
        "success": "" if not labels else outcome.get("success"),
        "failure_reason": outcome.get("failure_reason") or ("" if labels else "unlabeled"),
        "confidence": "" if confidence is None else round(confidence, 2),
        **{f"t_{phase}": round(phase_time[phase], 2) if labels else "" for phase in PHASES},
        "regrasps": outcome.get("regrasp_count", ""),
        # --grasp-assist episodes: "on (n)" with n welds engaged; "" before the flag existed.
        "assist": ("" if "grasp_assist" not in metadata else
                   f"on ({metadata.get('grasp_assist_activations', 0)})" if metadata["grasp_assist"] else "off"),
        "dropped_frames": "" if dropped is None else sum(dropped.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=DEFAULT_RECORD_DIR,
                        help="Episode folder (default: data/episodes)")
    parser.add_argument("--csv", help="Also write the table to this CSV file")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.directory).glob("*/episode.json")):
        try:
            rows.append(episode_row(json.loads(path.read_text())))
        except (OSError, ValueError, KeyError) as exc:
            print(f"{path.parent.name}: unreadable: {exc}", file=sys.stderr)
    if not rows:
        print("No episodes found.")
        return
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in COLUMNS}
    print("  ".join(column.ljust(widths[column]) for column in COLUMNS))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in COLUMNS))
    labeled = [row for row in rows if row["success"] != ""]
    print(f"{len(rows)} episodes, {len(labeled)} labeled, "
          f"{sum(row['success'] is True for row in labeled)} successful.")
    if args.csv:
        with open(args.csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
