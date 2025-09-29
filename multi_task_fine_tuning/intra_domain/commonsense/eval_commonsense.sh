#!/bin/sh

rank=8
alpha=16
gpuid=0
method=alora

model_p_or_n=meta-llama/Meta-Llama-3-8B

model_path=commonsense/models/$method-r$rank-a$alpha-3e4
results_path=commonsense/results/$method-r$rank-a$alpha-3e4
logs_path=commonsense/logs

mkdir -p $model_path
mkdir -p $results_path
mkdir -p $logs_path

# datasets: ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag

ds=ARC-Easy

CUDA_VISIBLE_DEVICES=$gpuid nohup python -u commonsense_evaluate.py \
    --model LLaMA3 \
    --adapter LoRA \
    --dataset $ds \
    --batch_size 1 \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path > $logs_path/$method-evaluate-$ds-r$rank-a$alpha-3e4.log 2>&1 &