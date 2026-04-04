#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/cl/replay.yaml}
shift $(( $# > 0 ? 1 : 0 ))

python cl_train.py --config "$CONFIG" "$@"
