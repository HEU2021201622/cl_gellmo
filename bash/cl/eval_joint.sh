#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/cl/eval_joint.yaml}
shift $(( $# > 0 ? 1 : 0 ))

LOCAL_ENV=${LOCAL_ENV:-evaluation}
TDC_ENV=${TDC_ENV:-pmo}

conda run -n "$LOCAL_ENV" python cl_eval_predict_local.py --config "$CONFIG" "$@"
conda run -n "$TDC_ENV" python cl_eval_predict_tdc.py --config "$CONFIG" "$@"
conda run -n "$LOCAL_ENV" python cl_eval.py --config "$CONFIG" "$@"
