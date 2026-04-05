#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/cl/infer_joint.yaml}
shift $(( $# > 0 ? 1 : 0 ))

python cl_infer.py --config "$CONFIG" "$@"
