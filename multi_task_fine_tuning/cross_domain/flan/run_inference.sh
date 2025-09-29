#!/bin/sh

rank=8
alpha=16
gpuid=0
method=alora

model_path=output/models/$method-r$rank-a$alpha-3e4
output_path=output/results/$method-r$rank-a$alpha-3e4
logs_path=output/logs/

mkdir -p $model_path
mkdir -p $output_path
mkdir -p $logs_path

CUDA_VISIBLE_DEVICES=$gpuid python inference.py \
      --base_model 'meta-llama/Llama-2-7b-hf' \
      --lora_weights_path "${model_path}/" \
      --lora_config_path "${model_path}/"   \
      --prompt_template "alpaca_short" \
      --output_file "${output_path}/inference.jsonl" \
      --test_file "./data/flan_test_200_selected_nstrict_1.jsonl" > $logs_path/$method-inference-r$rank-a$alpha-3e4.log 2>&1 &