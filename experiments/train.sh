#!/bin/bash

# Function to get the GPU model
get_gpu_model() {
  local gpu_info="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
  local gpu_model=""
  if [[ "$gpu_info" == *"GeForce RTX 3090"* ]]; then
    gpu_model="RTX3090"
  elif [[ "$gpu_info" == *"GeForce RTX 4090"* ]]; then
    gpu_model="RTX4090"
  elif [[ "$gpu_info" == *"GeForce GTX 1080 Ti"* ]]; then
    gpu_model="GTX1080Ti"
  else
    echo "Unknown GPU model: $gpu_info"
    exit 1
  fi
  echo "$gpu_model"
}


# Function to set batch size based on GPU model
set_batch_size() {
  local gpu_model="$1"
  case "$gpu_model" in
    "RTX3090")
      batch_size=16
      ;;
    "RTX4090")
      batch_size=8
      ;;
    "GTX1080Ti")
      batch_size=8
      ;;
    *)
      echo "Unknown GPU model: $gpu_model"
      exit 1
      ;;
  esac
}

# Check if all required parameters are provided
if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <dataset> <num_epochs> <lr_step> <lr> <task> <kd> <premodel>"
  exit 1
fi

# Get the parameters
dataset="$1"
num_epochs="$2"
lr_step="$3"
lr="$4"
task="$5"
kd="$6"
premodel="$7"

# Get the GPU model
gpu_model=$(get_gpu_model 0)  # Assuming only one GPU
set_batch_size "$gpu_model"
exp_name="${dataset}_${gpu_model}_epochs_${num_epochs}_lrstep_${lr_step}_lr_${lr}_bs_${batch_size}_task_${task}_kd_${kd}"


echo "Training on GPU: $gpu_model"
echo "Experiment Name: $exp_name"
echo "Batch Size: $batch_size"

# Run the training command with the updated parameters
# ADD --noshuffle flag to stop suffling for dataloader
cd src
python main.py "$task" \
    --exp_id "$exp_name" \
    --dataset "$dataset" \
    --freeze_components '{"base":         true,
                          "dla_up":       true,
                          "ida_up":       true,
                          "hm":           true,
                          "reg":          true,
                          "wh":           true,
                          "ltrb_amodal":  true,
                          "embedding":    false,
                          "tracking":     true}' \
    --optim adamw \
    --know_dist_weight "$kd" \
    --same_aug \
    --ltrb_amodal \
    --pre_hm \
    --hm_disturb 0.05 \
    --lost_disturb 0.4 \
    --fp_disturb 0.1 \
    --num_epochs "$num_epochs" \
    --val_intervals 5 \
    --lr_step "$lr_step" \
    --save_point "$lr_step" \
    --gpus 0 \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --num_workers 8 \
    --num_classes 1 \
    --embedding_loss focal \
    --multi_loss uncertainty \
    --load_model "$premodel"
cd ..
