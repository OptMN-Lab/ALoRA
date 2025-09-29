#!/bin/sh

rank=8
alpha=16
gpuid=0
method=alora

model_p_or_n=meta-llama/Meta-Llama-3-8B

model_path=math/models/$method-r$rank-a$alpha-3e4
results_path=math/results/$method-r$rank-a$alpha-3e4
logs_path=math/logs

mkdir -p $model_path
mkdir -p $results_path
mkdir -p $logs_path


CUDA_VISIBLE_DEVICES=$gpuid nohup python -u finetune.py \
  --base_model $model_p_or_n \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir $model_path \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r $rank \
  --lora_num 3 \
  --lora_alpha $alpha \
  --target_modules "["q_proj", "o_proj"]" > $logs_path/$method-finetune-r$rank-a$alpha-3e4.log 2>&1 &
