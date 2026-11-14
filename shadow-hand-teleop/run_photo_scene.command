#!/bin/zsh
set -e
cd "$(dirname "$0")"
export MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie"
scene_config="$PWD/assets/photo_scene_test/scene.json"
if [[ $# -gt 0 && "$1" != --* ]]; then
  scene_config="$1"
  shift
fi
echo "Opening the offline mesh-import feasibility scene. No Vision Pro required."
echo "Object provenance and physics assumptions are printed below."
scene_python="$PWD/.venv/bin/mjpython"
for scene_arg in "$@"; do
  if [[ "$scene_arg" == "--headless" ]]; then
    scene_python="$PWD/.venv/bin/python"
  fi
done
exec "$scene_python" "$PWD/scripts/view_dual_arms.py" --wrist human --workspace --scene-config "$scene_config" "$@"
