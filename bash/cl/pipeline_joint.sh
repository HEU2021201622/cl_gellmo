#!/bin/bash
set -euo pipefail

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

TRAIN_CONFIG=${1:-configs/cl/joint_all.yaml}
INFER_CONFIG=${2:-configs/cl/infer_joint.yaml}
EVAL_CONFIG=${3:-configs/cl/eval_joint.yaml}

bash bash/cl/train_joint_all.sh "$TRAIN_CONFIG"
bash bash/cl/infer_joint.sh "$INFER_CONFIG"
bash bash/cl/eval_joint.sh "$EVAL_CONFIG"
