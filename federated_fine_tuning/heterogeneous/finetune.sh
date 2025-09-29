#!/bin/sh

method=alora

model_path=output/models/$method-5e4
logs_path=output/logs/$method-5e4

mkdir -p $model_path
mkdir -p $logs_path

CUDA_VISIBLE_DEVICES=0 python main.py --global_model 'meta-llama/Llama-2-7b-hf'\
      --data_path  "./data/dataset1" \
      --output_dir  $model_path\
      --num_communication_rounds 11 \
      --num_clients  8 \
      --prompt_template_name "alpaca_short" \
      --client_selection_frac 1 \
      --local_num_epochs  10 \
      --local_batch_size  64 \
      --local_micro_batch_size 32 \
      --local_learning_rate 0.0005 \
      --lora_target_modules="[q_proj,v_proj]" > $logs_path/finetune-5e4.log 2>&1 &