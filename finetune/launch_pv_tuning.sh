#!/usr/bin/env bash
# PV-Tuning full fine-tuning launch script (runs inside tmux)
set -euo pipefail

cd /home/lxy/workspace/RiverOne

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec /home/lxy/miniconda3/envs/riverone/bin/python finetune/train_pv_tuning.py \
  --model_dir weights/RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP \
  --data_dir /home/lxy/workspace/datasets/vqa_format \
  --train_files train.jsonl \
  --output_dir finetune/outputs/pv_tuned_full_19k_3ep \
  --num_gpus 4 \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 4096 \
  --lr 3e-4 \
  --code_lr 3e-4 \
  --beam_size 3 \
  --max_code_change_per_step 1e-3 \
  --delta_decay 0.1 \
  --lr_scheduler cosine \
  --update_non_quantized_parameters \
  --no_freeze_vision \
  --gradient_checkpointing \
  --log_every_steps 10 \
  --save_every_steps 500 \
  2>&1 | tee finetune/outputs/pv_tuned_full_19k_3ep/train.log
