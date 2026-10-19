#!/bin/zsh
set -e
cd "$(dirname "$0")"
echo "Open Tracking Streamer, press Start, and keep the headset on."
read "address?Current local IP or room code: "
[[ -n "$address" ]] || exit 1
export MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie"
exec "$PWD/.venv/bin/mjpython" "$PWD/scripts/view_dual_arms.py" --wrist human --workspace --address "$address" "$@"
