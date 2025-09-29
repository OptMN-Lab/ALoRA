#!/bin/sh

method=alora

model_path=output/models/$method-5e4
output_path=output/results/$method-5e4
logs_path=output/logs/$method-5e4

mkdir -p $model_path
mkdir -p $output_path
mkdir -p $logs_path

client_id=0 # 0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=0 python GlobalModel_generated.py \
      --load_8bit \
      --base_model 'meta-llama/Llama-2-7b-hf' \
      --lora_weights_path "${model_path}/8/10/local_output_${client_id}/pytorch_model.bin" \
      --lora_config_path "${model_path}/8"   \
      --prompt_template "alpaca_short" \
      --output_file "${output_path}/inference_client${client_id}.jsonl" \
      --test_file "./data/dataset1/flan_test_200_selected_nstrict_1.jsonl" > $logs_path/inference_client$client_id.log 2>&1 &