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
