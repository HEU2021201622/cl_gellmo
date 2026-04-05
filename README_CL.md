# Continuous-Learning GeLLMO Handoff

## Overview

This repository extends the original GeLLMO training style with a continuous-learning training path while preserving the original design choices:

- pairwise lead optimization data format
- instruction-style prompts
- Mistral-7B-Instruct-v0.3 + LoRA
- original `train.sh` / `trainer.py` path kept intact

The new code now covers CL training, CL inference, and CL evaluation.

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
- `bash/cl/infer_sequence.sh`
- `bash/cl/infer_joint.sh`
- `bash/cl/eval_sequence.sh`
- `bash/cl/eval_joint.sh`
- `bash/cl/pipeline_sequence.sh`
- `bash/cl/pipeline_joint.sh`

### Modified existing files

- `trainer.py`
  - preserved original `train()` entrypoint
  - added reusable `train_on_records()` for CL
- `config.py`
  - added CL property aliases and defaults
- `cl_data.py`
  - now also supports normalized test-set filtering for CL inference

### New inference-side CL files

- `cl_infer.py`
  - main CL inference entrypoint
- `configs/cl/infer_sequence.yaml`
- `configs/cl/infer_joint.yaml`

### New evaluation-side CL files

- `cl_eval.py`
  - main CL evaluation entrypoint
- `cl_eval_props.py`
  - property prediction, canonicalization, similarity, diversity, and TDC parallel predictors
- `configs/cl/eval_sequence.yaml`
- `configs/cl/eval_joint.yaml`

## CL Experiment Definition

The staged CL setup is currently fixed to four steps:

1. `step1`: `drd2`
2. `step2`: `qed`
3. `step3`: `plogp`
4. `step4`: `jnk3`

Implemented training modes:

- `naive_sequence`
  - trains on current-step single-property data only
- `replay`
  - trains on current-step single-property data plus replay
- `joint_all`
  - one-shot sample-matched joint baseline over all four single-property tasks

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

By default, `lm_head` is not included in LoRA target modules. This avoids PEFT switching on extra embedding-layer saving during checkpoint export, which makes CL step-end saves heavier than necessary.

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

Checkpoint policy:

- intermediate `checkpoint-*` directories are for training recovery only
- at most 2 intermediate checkpoints are kept
- the step root directory is always kept as the final output

Inference outputs:

- `outputs/cl_infer/sequence/{method}/step{n}/{task}/{seen_or_unseen}/`
- `outputs/cl_infer/joint/{joint_name}/{run_name}/{task}/{seen_or_unseen}/`

Files per inference leaf directory:

- `predictions.json`
- `predictions.jsonl`
- `infer_manifest.json`

Each prediction record contains:

- `prompt`
- `response`
- `task`
- `split_type`
- `checkpoint_id`
- `checkpoint_dir`
- `source_smiles`

`response` is always a list and only contains model generations, never the prompt itself.

Evaluation outputs:

- `outputs/cl_eval/sequence/{method}/`
- `outputs/cl_eval/joint/{joint_name}/`

Files per evaluation root:

- `task_level_metrics.csv`
- `metrics_summary.csv`
- `cl_summary.csv`
- `seen_unseen_gap.csv`
- `best_candidates.jsonl`
- `eval_manifest.json`

## CL Inference

The CL inference path keeps the original GeLLMO prompt style:

- instruction-style prompts
- 0-shot
- beam search
- `num_beams=20`
- multiple candidates per input molecule

The current default test condition is:

- `seen`

### Sequence vs Joint loading

`sequence` mode:

- reads one adapter per step from `outputs/cl_train/{method}/step{n}`
- each step can infer any configured task list through `infer_tasks`
- tasks are not hard-coded in Python

`joint` mode:

- reads one adapter per configured run from `outputs/cl_train/{joint_name}/{run_name}`
- each run can infer any configured task list through `infer_tasks`

The important design choice is:

- training manifest describes what a checkpoint is
- inference config describes what tasks to test with that checkpoint

So later you can change task combinations only by editing YAML.

## Validation Status

### Confirmed on CPU

- `naive_sequence` dry-run passed through all four steps
- `replay` dry-run passed for:
  - full `step1 -> step2` replay path
  - `step3 -> step4` path and checkpoint chaining
- `joint_all` data loading path is valid, but dry-run may still be killed on the current machine due to resource limits

### Limitation

The current machine may kill long runs that repeatedly scan the full training table. Because of that:

- `naive_sequence` four-step dry-run is confirmed
- `replay` logic is confirmed, but a fresh full four-step replay dry-run should still be re-run on the server after any config change
- `joint_all` now uses per-task limits instead of consuming all available single-property records

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

### Sequence inference

```bash
bash bash/cl/infer_sequence.sh
```

or

```bash
python cl_infer.py --config configs/cl/infer_sequence.yaml
```

Example with a different sequence method:

```bash
python cl_infer.py \
  --config configs/cl/infer_sequence.yaml \
  --method replay \
  --output-root outputs/cl_infer/sequence/replay
```

### Joint inference

```bash
bash bash/cl/infer_joint.sh
```

or

```bash
python cl_infer.py --config configs/cl/infer_joint.yaml
```

### Dry-run inference

This resolves checkpoints, tasks, prompts, and output files without loading the model:

```bash
python cl_infer.py --config configs/cl/infer_sequence.yaml --dry-run
```

### Sequence evaluation

```bash
bash bash/cl/eval_sequence.sh
```

or

```bash
python cl_eval.py --config configs/cl/eval_sequence.yaml
```

Example for replay outputs:

```bash
python cl_eval.py \
  --config configs/cl/eval_sequence.yaml \
  --method replay \
  --input-root outputs/cl_infer/sequence/replay \
  --train-root outputs/cl_train/replay \
  --output-root outputs/cl_eval/sequence/replay
```

### Joint evaluation

```bash
bash bash/cl/eval_joint.sh
```

or

```bash
python cl_eval.py --config configs/cl/eval_joint.yaml
```

### One-shot pipeline

Sequence:

```bash
bash bash/cl/pipeline_sequence.sh naive_sequence
```

```bash
bash bash/cl/pipeline_sequence.sh replay
```

Joint:

```bash
bash bash/cl/pipeline_joint.sh
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


  ┌────────────┬────────────┬─────────────────┬─────────────────────┐                                                             
  │ 单属性 (4) │ 双属性 (6) │   三属性 (4)    │     四属性 (1)      │
  ├────────────┼────────────┼─────────────────┼─────────────────────┤                                                             
  │ drd2       │ drd2+plogp │ drd2+plogp+qed  │ jnk3+drd2+plogp+qed │
  ├────────────┼────────────┼─────────────────┼─────────────────────┤
  │ jnk3       │ drd2+qed   │ jnk3+drd2+plogp │                     │
  ├────────────┼────────────┼─────────────────┼─────────────────────┤
  │ plogp      │ jnk3+drd2  │ jnk3+drd2+qed   │                     │
  ├────────────┼────────────┼─────────────────┼─────────────────────┤
  │ qed        │ jnk3+plogp │ jnk3+plogp+qed  │                     │
  ├────────────┼────────────┼─────────────────┼─────────────────────┤
  │            │ jnk3+qed   │                 │                     │
  ├────────────┼────────────┼─────────────────┼─────────────────────┤
  │            │ plogp+qed  │                 │                     │
  └────────────┴────────────┴─────────────────┴─────────────────────┘