"""Send a test video (colour bars + clock) to the Vision Pro, without the simulation.

    .venv/bin/python scripts/check_headset_video.py <IP or room code>

If these bars show up in Tracking Streamer, video reaches the headset and any
problem is in the simulation's headset view; if not, check the app's video
settings and network. Ctrl-C stops.
"""
import argparse
import time

import cv2
import numpy as np
from avp_stream import VisionProStreamer

WIDTH, HEIGHT, FPS = 960, 540, 20
COLOURS = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
           (255, 0, 255), (0, 0, 255), (255, 0, 0)]   # BGR


def frame(t):
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    width = WIDTH // len(COLOURS)
    shift = int(t * 60) % WIDTH                       # bars scroll so motion is visible
    for i, colour in enumerate(COLOURS):
        image[:, i * width:(i + 1) * width] = colour
    image = np.roll(image, shift, axis=1)
    cv2.rectangle(image, (0, HEIGHT - 90), (WIDTH, HEIGHT), (0, 0, 0), -1)
    cv2.putText(image, f"lab3l video test  {time.strftime('%H:%M:%S')}", (30, HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address", help="Vision Pro IP (same Wi-Fi) or room code")
    args = parser.parse_args()
    streamer = VisionProStreamer(ip=args.address, record=False, benchmark_quiet=True, verbose=True)
    streamer.configure_video(device=None, size=f"{WIDTH}x{HEIGHT}", fps=FPS, stereo=False)
    streamer.start_webrtc(blocking=False)
    print("Sending test video. Look for colour bars in Tracking Streamer. Ctrl-C stops.", flush=True)
    start, last_report, sent = time.monotonic(), 0., 0
    try:
        while True:
            now = time.monotonic() - start
            streamer.update_frame(frame(now))         # a new array every frame
            sent += 1
            if now - last_report >= 2:
                last_report = now
                # is_connected() can stay False while video flows (ICE "completed");
                # the media clock starts when the headset first pulls a frame.
                receiving = getattr(getattr(streamer, "_media_clock", None), "is_started", False)
                print(f"{now:5.0f} s  headset receiving video={bool(receiving)}  frames sent={sent}",
                      flush=True)
            time.sleep(1 / FPS)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            streamer.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
