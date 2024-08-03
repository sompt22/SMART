#!/bin/bash

# This script runs inference on a video file
model_path=$(realpath $1)


cd src
python test.py tracking,embedding --exp_id sompt22-max150-noprehm-kd08-simscore065 \
                                  --dataset sompt22 \
                                  --trainval \
                                  --ltrb_amodal \
                                  --max_age 150 \
                                  --track_thresh 0.4 \
                                  --pre_thresh 0.5 \
                                  --load_model $model_path \
                                  --num_classes 1
cd ..