#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/cl/eval_joint.yaml}
shift $(( $# > 0 ? 1 : 0 ))

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/bin/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/bin/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/workspace/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/workspace/miniconda3/etc/profile.d/conda.sh"
  else
    echo "conda command not found and no known conda.sh init script was found." >&2
    exit 1
  fi
fi

LOCAL_ENV=${LOCAL_ENV:-evaluation}
TDC_ENV=${TDC_ENV:-pmo}

conda run -n "$LOCAL_ENV" python cl_eval_predict_local.py --config "$CONFIG" "$@"
conda run -n "$TDC_ENV" python cl_eval_predict_tdc.py --config "$CONFIG" "$@"
conda run -n "$LOCAL_ENV" python cl_eval.py --config "$CONFIG" "$@"
