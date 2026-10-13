# Holos Vision Pro Teleoperation Proof of Concept

This project is the local simulation workspace for testing this pipeline:

```text
Apple Vision Pro hand tracking
        -> VisionProTeleop / avp_stream
        -> MuJoCo Shadow Hand
        -> robot-state and action demonstrations
        -> later policy training
```

## Current status

- MuJoCo 3.15 is installed in the local `.venv`.
- MuJoCo Menagerie and the Shadow Hand model are installed and cached locally.
- `avp_stream` is installed for the later Vision Pro connection.
- The official VisionProTeleop examples are available under `third_party/VisionProTeleop`.
- A simulation smoke test and a local Shadow Hand viewer are ready.

## Open the Shadow Hand simulation

Double-click `run_shadow_hand.command` in Finder. If macOS asks whether to open it,
choose **Open**.

The default launcher opens the left hand. To open the right-hand model instead,
double-click `run_right_shadow_hand.command`. Click inside either MuJoCo viewer,
then press **F** to make a fist or **O** to open the hand.

To show both hands in the same MuJoCo window, run
`run_dual_shadow_hands.command`. Its keyboard controls are:

- **1**: toggle the left hand between open and fist
- **2**: toggle the right hand between open and fist
- **F**: make both hands into fists
- **O**: open both hands

The same action can be run from Terminal:

```bash
./run_shadow_hand.command
```

## Live two-hand finger control (first prototype)

Run `run_live_dual_hands.command` in Finder and enter the CURRENT Tracking Streamer
IP or remote room code. Keep the headset and app active. A MuJoCo window on the
Mac shows both fixed Shadow Hands; this launcher does not stream the scene into
the headset. SPACE pauses/resumes; closing the window disconnects. No recordings,
camera input or microphone input are enabled. Remote mode uses third-party
signaling and, when needed, an encrypted TURN relay.

This first retargeter maps individual finger bend angles, clips joint targets,
and rate-limits commands. It does NOT reproduce the complete human hand pose:
spread, thumb opposition, wrist motion, and arm reaching are not implemented.
No fresh packets for 0.5 seconds freezes the full simulation if both hands are
stale; a single stale hand holds its last actuator target. The SDK does not expose
per-hand tracking validity in this interface, so a frozen but repeatedly sent
skeleton cannot always be distinguished from a deliberately still hand.
This is a connection/control prototype, not a calibrated grasping controller or
a trained policy. Two robot arms have not been added yet.

Offline checks: `.venv/bin/python scripts/test_hand_retargeting.py`.

## Mock teleoperation/data-collection demo

Before the Vision Pro is available, double-click `run_mock_demo.command`. The
simulated hand performs an open-close motion and saves a demonstration to:

```text
data/mock_grasp_demo.npz
data/mock_grasp_demo.json
```

The dataset contains timestamps, robot positions and velocities, commanded
actions, the object's pose, and the number of active contacts. This is a pipeline
test rather than a successful-task label; the outcome remains `not_evaluated`.

## Verify the local installation

```bash
MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie" \
  .venv/bin/python scripts/check_setup.py
```

## Connect Apple Vision Pro later

1. Install and open **Tracking Streamer** on Apple Vision Pro.
2. Put the Vision Pro and this Mac on the same Wi-Fi.
3. Press **Start** in Tracking Streamer and note the displayed IP address.
4. Run:

```bash
.venv/bin/python scripts/check_vision_pro.py VISION_PRO_IP
```

Replace `VISION_PRO_IP` with the address shown in the headset. A room code can be
used instead when using VisionProTeleop's external-network mode.

After the basic stream works, the official Shadow Hand controller is located at:

```text
third_party/VisionProTeleop/examples/12_diffik_shadow_hand.py
```

For the first live test, you can instead double-click
`run_vision_pro_shadow_hand.command`. It asks for the IP address or room code and
then launches the official VisionProTeleop Shadow Hand controller.

## Planned milestones

1. Confirm live Vision Pro hand tracking on the Mac.
2. Replace the mock input with the existing VisionProTeleop Shadow Hand controller.
3. Record one real-time grasping demonstration with a fixed Shadow Hand.
4. Define one household manipulation task and evaluation metric.
5. Add a simulated robotic arm only after fixed-hand control is reliable.
