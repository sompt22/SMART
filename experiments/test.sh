#!/bin/bash

# This script runs inference on a video file
model_path=$(realpath $1)


cd src
python test.py tracking,embedding --exp_id mot17-divo-max150-noprehm-kd0-simscore065 \
                                  --dataset mot17 \
                                  --trainval \
                                  --ltrb_amodal \
                                  --max_age 150 \
                                  --track_thresh 0.4 \
                                  --pre_thresh 0.5 \
                                  --load_model $model_path \
                                  --num_classes 1
cd ..