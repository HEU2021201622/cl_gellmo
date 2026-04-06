#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/cl/eval_sequence.yaml}
shift $(( $# > 0 ? 1 : 0 ))

source ~/.bashrc
eval "$(conda shell.bash hook)"

LOCAL_ENV=${LOCAL_ENV:-evaluate}
TDC_ENV=${TDC_ENV:-pmo}

conda activate "$LOCAL_ENV"
python cl_eval_predict_local.py --config "$CONFIG" "$@"

conda activate "$TDC_ENV"
python cl_eval_predict_tdc.py --config "$CONFIG" "$@"

conda activate "$LOCAL_ENV"
python cl_eval.py --config "$CONFIG" "$@"
