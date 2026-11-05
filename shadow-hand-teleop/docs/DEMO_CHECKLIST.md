# lab3l Studio demo checklist

## Before

- [ ] Charge the Vision Pro, and keep its charger nearby.
- [ ] Put the Mac and the headset on the same Wi-Fi.
- [ ] Close every other simulation (the MuJoCo windows, replays, benchmarks).
- [ ] Double-click `run_studio.command`. The page opens at `http://127.0.0.1:8765`.
- [ ] Put on the headset, open Tracking Streamer, press Start, and enter its
      IP or room code on the page. Then Start.
- [ ] Record one test attempt end to end and check the result card plays.
- [ ] Pick a fallback episode with a clear SUCCESS (for example 0016) and
      note its number.
- [ ] Turn off Mac notifications. Set the projector to mirror or extend so
      the page is readable from the back.

## During

1. **Start** (the form folds away; both chips should turn amber, then green).
2. **① Reset** if highlighted, so the cup is at its start.
3. **② Calibrate**: face forward, hands comfortably apart.
4. **③ Record**, do the attempt, press **Record** again to stop.
5. The result plays: outcome badge, video and phase timeline.

## If a grasp fails

Show the failure label: the system telling you why it failed is a feature.
Then click the good fallback episode in the list on the right.

## If tracking drops

A chip turns amber (or grey): wait for it to come back, then **Calibrate**.
With auto-recenter on, short gaps re-anchor by themselves.

## If the sim crashes

The log opens with the exit code. Press **Start** again; recorded episodes
are already saved. Keep demoing from the episode list if needed.
