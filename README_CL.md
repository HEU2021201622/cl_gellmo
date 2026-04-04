# Continuous-Learning GeLLMO Handoff

## Overview

This repository extends the original GeLLMO training style with a continuous-learning training path while preserving the original design choices:

- pairwise lead optimization data format
- instruction-style prompts
- Mistral-7B-Instruct-v0.3 + LoRA
- original `train.sh` / `trainer.py` path kept intact

The new code currently focuses on training only. Inference and evaluation for CL are not implemented yet, but the output structure was designed so later stages can reuse it directly.

## What Was Added

### New training-side CL files

- `cl_train.py`
  - main CL training entrypoint
- `cl_data.py`
  - name-based data loading and step dataset construction
- `cl_buffer.py`
  - replay buffer implementation
- `cl_config.py`
  - YAML config loader for `configs/cl/*.yaml`
- `cl_manifest.py`
  - step-level manifest writing

### Configs

- `configs/cl/base.yaml`
- `configs/cl/naive_sequence.yaml`
- `configs/cl/replay.yaml`
- `configs/cl/joint_all.yaml`

### Bash entrypoints

- `bash/cl/train_naive_sequence.sh`
- `bash/cl/train_replay.sh`
- `bash/cl/train_joint_all.sh`

### Modified existing files

- `trainer.py`
  - preserved original `train()` entrypoint
  - added reusable `train_on_records()` for CL
- `config.py`
  - added CL property aliases and defaults

## CL Experiment Definition

The staged CL setup is currently fixed to four steps:

1. `step1`: `drd2`
2. `step2`: `qed` + combo calibration `drd2+qed`
3. `step3`: `plogp` + combo calibration `drd2+qed+plogp`
4. `step4`: `jnk3` + combo calibration `drd2+qed+plogp+jnk3`

Implemented training modes:

- `naive_sequence`
  - trains on `new + combo`
- `replay`
  - trains on `new + combo + replay`
- `joint_all`
  - one-shot joint upper-bound baseline

## Data Assumptions

Current CL training reads from:

- `data/all_data/train.jsonl`

Expected pairwise fields:

- `source_smiles`
- `target_smiles`
- `task`
- `properties`

Optional but supported:

- `instruction`
- `instr_idx`
- `instr_setting`
- `scaffold`
- `similarity`

Important rules:

- fields are parsed by name, not by column order
- property aliases are normalized
- `logp` is normalized to `plogp`

## LoRA Target Modules

LoRA injection is still done through the shared training logic in `trainer.py`.

Current behavior:

- if `lora_target_modules` is explicitly provided, that value is used
- otherwise the code auto-selects target modules from the model name / path

Currently supported model families:

- `mistral`
- `llama`

The default target modules for both are currently:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

This means the server-side local Llama path below is expected to be auto-detected as a `llama` model:

```bash
/home/xy/workspace/llm/Llama-3-8-ins
```

If needed, you can still override manually:

```bash
python cl_train.py \
  --config configs/cl/naive_sequence.yaml \
  --base-model /home/xy/workspace/llm/Llama-3-8-ins \
  --method naive_sequence
```

If future models use different internal layer names, update the family mapping in `config.py`.

## Output Layout

Training outputs:

- `outputs/cl_train/{method}/step{n}/...`
- `outputs/cl_train/joint_all/all/...`

Examples:

- `outputs/cl_train/naive_sequence/step1/`
- `outputs/cl_train/replay/step4/`
- `outputs/cl_train/joint_all/all/`

Each step directory is designed to be reusable later by infer/eval code and includes:

- `train_manifest.json`
- `train_config.resolved.yaml`
- LoRA checkpoint files after real training
- optional `step_train_records.jsonl` and `step_val_records.jsonl`
- replay metadata files for replay runs after real training

## Validation Status

### Confirmed on CPU

- `naive_sequence` dry-run passed through all four steps
- `replay` dry-run passed for:
  - full `step1 -> step2` replay path
  - `step3 -> step4` path and checkpoint chaining
- `joint_all` dry-run succeeded at least once earlier

### Limitation

The current machine may kill long runs that repeatedly scan the full training table. Because of that:

- `naive_sequence` four-step dry-run is confirmed
- `replay` logic is confirmed, but a full one-shot four-step replay dry-run should be re-run on the server

Recommended first check on the server:

```bash
python cl_train.py --config configs/cl/replay.yaml --dry-run
```

## Environment Check

Before training, verify imports:

```bash
python - <<'PY'
import torch, transformers, datasets, peft, fire
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("imports ok")
PY
```

CPU-only dry-run:

```bash
python cl_train.py --config configs/cl/naive_sequence.yaml --dry-run
```

## How To Run

### Naive sequence

```bash
bash bash/cl/train_naive_sequence.sh
```

or

```bash
python cl_train.py --config configs/cl/naive_sequence.yaml
```

### Replay

```bash
bash bash/cl/train_replay.sh
```

or

```bash
python cl_train.py --config configs/cl/replay.yaml
```

### Joint baseline

```bash
bash bash/cl/train_joint_all.sh
```

or

```bash
python cl_train.py --config configs/cl/joint_all.yaml
```

### Useful overrides

Subset of steps:

```bash
python cl_train.py \
  --config configs/cl/naive_sequence.yaml \
  --start-step step1 \
  --end-step step2
```

Resume from an adapter:

```bash
python cl_train.py \
  --config configs/cl/replay.yaml \
  --start-step step3 \
  --end-step step4 \
  --resume-adapter outputs/cl_train/replay/step2
```

Dry-run:

```bash
python cl_train.py --config configs/cl/replay.yaml --dry-run
```

Override replay settings:

```bash
python cl_train.py \
  --config configs/cl/replay.yaml \
  --buffer-capacity 5000 \
  --replay-ratio 0.2
```

## Suggested Next Steps On The Server

1. Verify environment imports
2. Run `python cl_train.py --config configs/cl/replay.yaml --dry-run`
3. Inspect `outputs/cl_train/replay/step*/train_manifest.json`
4. Start real training once dry-run looks correct
5. Implement or wire CL inference/evaluation later
