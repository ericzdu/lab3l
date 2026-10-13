#!/bin/zsh
set -e

cd "$(dirname "$0")"

echo "Enter the IP address shown by Tracking Streamer on Apple Vision Pro"
echo "(or enter its external-network room code):"
read "address?> "

if [[ -z "$address" ]]; then
  echo "No address was entered. Nothing was started."
  exit 1
fi

exec "$PWD/.venv/bin/python" \
  "$PWD/third_party/VisionProTeleop/examples/12_diffik_shadow_hand.py" \
  --ip "$address"

