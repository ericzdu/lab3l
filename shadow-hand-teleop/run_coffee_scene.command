#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -f "$PWD/local_scene_assets/coffee_machine_v1/scene.json" ]]; then
  echo "The locally generated coffee-machine bundle is missing."
  echo "See PHOTO_SCENE_SETUP.md to prepare a downloaded GLB, or run run_photo_scene.command for the synthetic test."
  exit 1
fi
exec /bin/zsh "$PWD/run_photo_scene.command" "$PWD/local_scene_assets/coffee_machine_v1/scene.json" "$@"
