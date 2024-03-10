#!/bin/bash

# This script runs inference on a video file
model_path=$(realpath $1)


cd src
python test.py tracking --exp_id sompt22-train-sim --dataset sompt22-train-sim --trainval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model $model_path --num_classes 1
cd ..