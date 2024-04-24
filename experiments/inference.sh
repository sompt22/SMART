#!/bin/bash

# This script runs inference on a video file

model_path=$(realpath $1)
video_path=$(realpath $2)
task=$3
debug=$4

echo "$video_path"
echo "$model_path"
echo FATIH
# Use sed to extract the required path
model_path_=$(echo "$model_path" | sed -n "s|.*/${task}/[^/]*/\(.*\)/model_last.pth.*|\1|p" | sed 's|/logs_|_logs_|')
echo "$model_path_"
echo EMRE

video_path_=$(basename "$video_path" | cut -f1 -d'.')
experiment_name="inference_${video_path_}_${model_path_}"
printf "experiment_name: $experiment_name\n"

cd src
# infer --motchallenge saves inference results
python demo.py $task --exp_id $experiment_name \
                     --ltrb_amodal \
                     --max_age 15 \
                     --debug $debug \
                     --load_model $model_path \
                     --track_thresh 0.4 \
                     --pre_thresh 0.5 \
                     --num_classes 1 \
                     --demo $video_path 
cd ..
