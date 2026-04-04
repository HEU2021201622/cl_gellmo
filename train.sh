#!/bin/bash

# Usage: train.sh <model> <data_dir> <model_dir> <expt_dir> <num_epochs> <tasks> <resume_from_checkpoint>

base_model=$1
data_dir=$2
expt_dir=$3
num_epochs=$4
tasks=$5
opt_type="simple"
resume_from_checkpoint=$7

model_lower=$(echo $model | tr '[:upper:]' '[:lower:]')
#base_model="${model_dir}/${base_model}/"
lora_target_modules="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head']"

#if resume_from_checkpoint is not None, then the model will be loaded from the checkpoint, else set to False
if [[ $resume_from_checkpoint == "" ]]; then
   resume_from_checkpoint=False
fi

if [[ $model == "Llama-3.1-70B-Instruct" ]]; then
   python trainer.py \
      --data_path $data_dir \
      --base_model $base_model \
      --lora_target_modules $lora_target_modules \
      --output_dir $expt_dir \
      --num_epochs $num_epochs \
      --opt_type $opt_type \
      --train_set_size 10000 \
      --resume_from_checkpoint $resume_from_checkpoint \
      --tasks $tasks \
      --load_in_8bit
else
   python trainer.py \
      --data_path $data_dir \
      --base_model $base_model \
      --lora_target_modules $lora_target_modules \
      --output_dir $expt_dir \
      --num_epochs $num_epochs \
      --opt_type $opt_type \
      --train_set_size 10000 \
      --resume_from_checkpoint $resume_from_checkpoint \
      --tasks $tasks
fi