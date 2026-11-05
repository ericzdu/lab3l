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
# Silence gRPC fork/poll INFO lines (fork_posix.cc, ev_poll_posix.cc); errors still print.
export GRPC_VERBOSITY=ERROR GLOG_minloglevel=2
exec "$PWD/.venv/bin/mjpython" "$PWD/scripts/view_dual_arms.py" --address "$address" "$@"
