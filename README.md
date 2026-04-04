# GeLLMO

This repository contains the GeLLMO molecule optimization codebase together with an incremental continuous-learning training path built on top of the original project structure.

## Main Entry Docs

- Original-style project usage:
  - training: `train.sh`
  - inference: `inference.sh`
  - evaluation notebooks: `process-output.ipynb`, `evaluate.ipynb`
- Continuous-learning handoff and usage:
  - see [README_CL.md](/home/xy/projects/CL_gellmo/GeLLMO/README_CL.md)

## Current CL Training Entrypoints

- `bash/cl/train_naive_sequence.sh`
- `bash/cl/train_replay.sh`
- `bash/cl/train_joint_all.sh`

## Output Convention

- training: `outputs/cl_train/{method}/step{n}/...`
- joint baseline: `outputs/cl_train/joint_all/all/...`

For the full CL setup, validation status, and server handoff instructions, read [README_CL.md](/home/xy/projects/CL_gellmo/GeLLMO/README_CL.md).
