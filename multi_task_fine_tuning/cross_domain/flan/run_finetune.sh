#!/bin/sh

rank=8
alpha=16
gpuid=0
method=alora

model_p_or_n=meta-llama/Llama-2-7b-hf

model_path=output/models/$method-r$rank-a$alpha-3e4
logs_path=output/logs/

mkdir -p $model_path
mkdir -p $logs_path


CUDA_VISIBLE_DEVICES=$gpuid nohup python -u finetune.py \
  --base_model $model_p_or_n \
  --data_path "data/flan_multi_task_data.json" \
  --output_dir $model_path \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 50 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r $rank \
  --lora_num 3 \
  --lora_alpha $alpha \
  --target_modules "["q_proj", "v_proj"]" > $logs_path/$method-finetune-r$rank-a$alpha-3e4.log 2>&1 &
