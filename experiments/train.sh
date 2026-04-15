#!/bin/bash

set -euo pipefail

# Check if all required parameters are provided
if [ "$#" -lt 9 ] || [ "$#" -gt 11 ]; then
  echo "Usage: $0 <dataset> <num_epochs> <lr_step> <lr> <task> <kd> <loss> <optim> <premodel> [batch_size] [gpus]"
  exit 1
fi

# Get the parameters
dataset="$1"
num_epochs="$2"
lr_step="$3"
lr="$4"
task="$5"
kd="$6"
loss="$7"
optim="$8"
premodel="$9"
batch_size="${10:-8}"
gpus="${11:-0}"

device_tag="gpus_${gpus//,/x}"
exp_name="${dataset}_${device_tag}_epochs_${num_epochs}_lrstep_${lr_step}_lr_${lr}_bs_${batch_size}_kd_${kd}_loss_${loss}_optim_${optim}"

echo "Experiment Name: $exp_name"
echo "Batch Size: $batch_size"
echo "GPUs: $gpus"

# Run the training command with the updated parameters
cd src
python3 main.py "$task" \
    --exp_id "$exp_name" \
    --dataset "$dataset" \
    --freeze_components '{"base":         false,
                          "dla_up":       false,
                          "ida_up":       false,
                          "hm":           false,
                          "reg":          false,
                          "wh":           false,
                          "ltrb_amodal":  false,
                          "embedding":    false,
                          "tracking":     false}' \
    --optim "$optim" \
    --know_dist_weight "$kd" \
    --same_aug_pre \
    --ltrb_amodal \
    --pre_hm \
    --hm_disturb 0.05 \
    --lost_disturb 0.4 \
    --fp_disturb 0.1 \
    --num_epochs "$num_epochs" \
    --val_intervals 5 \
    --lr_step "$lr_step" \
    --save_point "$lr_step" \
    --gpus "$gpus" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --num_workers 4 \
    --num_classes 1 \
    --embedding_loss "$loss" \
    --multi_loss uncertainty \
    --load_model "$premodel"
cd ..
