#!/bin/bash

# This script runs inference on a video file

model_path=$(realpath $1)
video_path=$(realpath $2)
task=$3
debug=$4

echo "$video_path"
echo "$model_path"

_experiment_name=$(basename "$video_path" | cut -f1 -d'.')
experiment_name='inference_'$_experiment_name
printf "experiment_name: $experiment_name\n"

cd src
# infer --motchallenge saves inference results
python demo.py $task --exp_id $experiment_name --ltrb_amodal --pre_hm --max_age 15 --debug $debug --load_model $model_path --num_classes 1 --demo $video_path 
cd ..
