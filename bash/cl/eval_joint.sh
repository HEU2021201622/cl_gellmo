#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/cl/eval_joint.yaml}
shift $(( $# > 0 ? 1 : 0 ))

python cl_eval.py --config "$CONFIG" "$@"
