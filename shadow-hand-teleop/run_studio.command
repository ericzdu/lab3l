#!/bin/zsh
# lab3l Studio: local web page for recording sessions and results (127.0.0.1:8765).
set -e
cd "$(dirname "$0")"
export MENAGERIE_CACHE_DIR="$PWD/.cache/mujoco_menagerie"
# Silence gRPC fork/poll INFO lines in the sim Studio starts; errors still print.
export GRPC_VERBOSITY=ERROR GLOG_minloglevel=2
cd scripts
exec ../.venv/bin/python -m studio.server --port 8765 --open "$@"
