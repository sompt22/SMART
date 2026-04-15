#!/bin/bash

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  echo "Usage: $0 <model_path> <video_path> <task> <debug> [dataset] [display]"
  exit 1
fi

model_path="$(realpath "$1")"
video_path="$(realpath "$2")"
task="$3"
debug="$4"
dataset="${5:-mot17}"
display="${6:-0}"

model_path_=$(echo "$model_path" | sed -n "s|.*/${task}/[^/]*/\(.*\)/model_last.pth.*|\1|p" | sed 's|/logs_|_logs_|')
if [ -z "$model_path_" ]; then
  model_path_="$(basename "$(dirname "$model_path")")"
fi

video_path_=$(basename "$video_path" | cut -f1 -d'.')
experiment_name="inference_${video_path_}_${model_path_}"
printf "experiment_name: $experiment_name\n"

display_flag=()
if [ "$display" = "1" ]; then
  display_flag=(--display)
fi

cd src
python3 demo.py "$task" --exp_id "$experiment_name" \
                        --dataset "$dataset" \
                        --ltrb_amodal \
                        --max_age 50 \
                        --debug "$debug" \
                        --load_model "$model_path" \
                        --track_thresh 0.4 \
                        --pre_thresh 0.5 \
                        --num_classes 1 \
                        --demo "$video_path" \
                        "${display_flag[@]}"
cd ..
