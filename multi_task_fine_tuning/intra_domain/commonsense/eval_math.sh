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

# datasets: AQuA gsm8k SVAMP mawps (SingleEq)

ds=AQuA

CUDA_VISIBLE_DEVICES=$gpuid nohup python -u math_evaluate.py \
    --model LLaMA3 \
    --adapter LoRA \
    --dataset $ds \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path > $logs_path/$method-evaluate-$ds-r$rank-a$alpha-3e4.log 2>&1 &