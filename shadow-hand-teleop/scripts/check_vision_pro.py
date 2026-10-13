"""Print live Apple Vision Pro hand-tracking samples."""

from __future__ import annotations

import argparse
import time

from avp_stream import VisionProStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to Tracking Streamer and print live hand data."
    )
    parser.add_argument(
        "address",
        help="Vision Pro IP address on the same Wi-Fi, or its external room code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    streamer = VisionProStreamer(
        ip=args.address,
        record=False,
        benchmark_quiet=True,
        verbose=True,
    )

    print("Waiting for tracking data. Press Control-C to stop.")
    try:
        while True:
            sample = streamer.get_latest(use_cache=True)
            if sample is not None:
                wrist = sample.right.wrist[:3, 3]
                pinch = sample.right.pinch_distance
                print(
                    "right wrist "
                    f"x={wrist[0]: .3f}, y={wrist[1]: .3f}, z={wrist[2]: .3f}; "
                    f"pinch={pinch:.3f} m"
                )
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("Stopping Vision Pro connection.")
    finally:
        streamer.cleanup()


if __name__ == "__main__":
    main()

