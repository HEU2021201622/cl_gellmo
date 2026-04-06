#!/bin/bash
set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"

METHOD=${1:-naive_sequence}
TRAIN_CONFIG=${2:-configs/cl/${METHOD}.yaml}
INFER_CONFIG=${3:-configs/cl/infer_sequence.yaml}
EVAL_CONFIG=${4:-configs/cl/eval_sequence.yaml}

if [[ "$METHOD" == "replay" ]]; then
  bash bash/cl/train_replay.sh "$TRAIN_CONFIG"
else
  bash bash/cl/train_naive_sequence.sh "$TRAIN_CONFIG"
fi

python cl_infer.py \
  --config "$INFER_CONFIG" \
  --method "$METHOD" \
  --output-root "outputs/cl_infer/sequence/${METHOD}"

python cl_eval.py \
  --config "$EVAL_CONFIG" \
  --method "$METHOD" \
  --input-root "outputs/cl_infer/sequence/${METHOD}" \
  --train-root "outputs/cl_train/${METHOD}" \
  --output-root "outputs/cl_eval/sequence/${METHOD}"
