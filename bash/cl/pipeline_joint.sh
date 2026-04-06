#!/bin/bash
set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"

TRAIN_CONFIG=${1:-configs/cl/joint_all.yaml}
INFER_CONFIG=${2:-configs/cl/infer_joint.yaml}
EVAL_CONFIG=${3:-configs/cl/eval_joint.yaml}

bash bash/cl/train_joint_all.sh "$TRAIN_CONFIG"
bash bash/cl/infer_joint.sh "$INFER_CONFIG"
bash bash/cl/eval_joint.sh "$EVAL_CONFIG"
