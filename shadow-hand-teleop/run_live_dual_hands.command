#!/bin/zsh
set -e
cd "$(dirname "$0")"
echo "Open Tracking Streamer and keep the headset on."
echo "Enter its CURRENT local IP or remote room code:"
read "address?> "
if [[ -z "$address" ]]; then
  echo "No address entered. Nothing started."
  exit 1
fi
export MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie"
exec "$PWD/.venv/bin/mjpython" "$PWD/scripts/teleop_shadow_hands.py" "$address"
