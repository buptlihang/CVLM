#!/bin/bash

model_path=$1  # The path to the checkpoint
image_path=$2
CUDA_VISIBLE_DEVICES=0  python -m evaluation.MME.evaluate \
    --model-path $model_path \
    --question-file ./evaluation/MME/mme.jsonl \
    --image-folder $image_path \
    --answers-dir ./evaluation/MME/results \
    --temperature 0
python -m evaluation.MME.calculation --results_dir evaluation/MME/results/

