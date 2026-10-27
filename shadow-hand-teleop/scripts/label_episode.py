"""Label recorded episodes (success + phases) with episode_labels.py.

Accepts episode.h5 files, episode folders, or a folder of episode folders.
Labels are heuristic; see episode_labels.LabelConfig for every threshold.
"""
import argparse
from pathlib import Path
import sys

from episode_labels import label_episode, load_episode, summary_line, write_labels


def episode_files(paths):
    for path in map(Path, paths):
        if path.is_dir() and not (path / "episode.h5").exists():
            yield from sorted(path.glob("*/episode.h5"))
        else:
            yield path / "episode.h5" if path.is_dir() else path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="episode.h5 files or episode folders")
    parser.add_argument("--dry-run", action="store_true", help="Print summaries without writing")
    args = parser.parse_args()
    failures = 0
    files = list(episode_files(args.paths))
    if not files:
        parser.error("No episode.h5 files found")
    for path in files:
        try:
            arrays, metadata = load_episode(path)
            labels = label_episode(arrays, metadata)
            if not args.dry_run:
                write_labels(path, labels)
            print(summary_line(metadata["episode_id"], labels))
        except (OSError, ValueError, KeyError) as exc:
            failures += 1
            print(f"{path.parent.name}: could not label: {exc}", file=sys.stderr)
    if args.dry_run:
        print("Dry run: nothing written.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
