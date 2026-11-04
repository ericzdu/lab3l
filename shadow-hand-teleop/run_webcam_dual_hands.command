#!/bin/zsh
set -e
cd "$(dirname "$0")"
echo "Starting webcam hand tracking. macOS may ask for camera permission."
export MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie"
exec "$PWD/.venv/bin/mjpython" "$PWD/scripts/webcam_shadow_hands.py" "$@"
