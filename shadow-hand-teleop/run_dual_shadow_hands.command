#!/bin/zsh
set -e

cd "$(dirname "$0")"
export MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie"
exec "$PWD/.venv/bin/mjpython" "$PWD/scripts/view_dual_shadow_hands.py"
