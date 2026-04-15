#!/bin/bash
# ============================================================
# SMART Docker Entrypoint
#
# Usage:
#   train  <task> [main.py args...]
#   infer  <task> [demo.py args...]
#   test   <task> [test.py args...]
#   shell  — drop into bash
# ============================================================
set -e

WORKDIR=/workspace/SMART/src
cd "$WORKDIR"

CMD="$1"
shift || true

case "$CMD" in

  train)
    echo "[SMART] Data dir: ${SMART_DATA_DIR:-/workspace/SMART/data}"
    echo "[SMART] Starting training: python main.py $*"
    exec python main.py "$@"
    ;;

  infer)
    echo "[SMART] Data dir: ${SMART_DATA_DIR:-/workspace/SMART/data}"
    echo "[SMART] Starting inference: python demo.py $*"
    exec python demo.py "$@"
    ;;

  test)
    echo "[SMART] Data dir: ${SMART_DATA_DIR:-/workspace/SMART/data}"
    echo "[SMART] Starting evaluation: python test.py $*"
    exec python test.py "$@"
    ;;

  unittest)
    echo "[SMART] Running unit tests..."
    cd /workspace/SMART
    python test_association.py   -v
    python test_model_training.py -v
    ;;

  shell)
    exec /bin/bash
    ;;

  --help | help | "")
    cat << 'EOF'
SMART Multi-Object Tracker — Docker Image

Commands:
  train  <task> [args]   Run src/main.py  (training loop)
  infer  <task> [args]   Run src/demo.py  (video inference)
  test   <task> [args]   Run src/test.py  (benchmark evaluation)
  unittest               Run test_association + test_model_training
  shell                  Open bash shell

Tasks:  tracking | embedding | tracking,embedding

Training example:
  docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/exp:/workspace/SMART/exp \
    smart train tracking,embedding \
      --exp_id mot17_baseline \
      --dataset mot17 \
      --num_epochs 30 \
      --lr 2.5e-4 \
      --lr_step 20 \
      --batch_size 8 \
      --num_workers 4 \
      --gpus 0 \
      --num_classes 1 \
      --multi_loss uncertainty \
      --load_model /data/pretrained.pth

Inference example:
  docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/exp:/workspace/SMART/exp \
    smart infer tracking,embedding \
      --exp_id my_inference \
      --load_model /data/model_last.pth \
      --demo /data/video.mp4 \
      --track_thresh 0.4 \
      --num_classes 1

Mount points:
  /data                Dataset root, pretrained weights, videos
  /workspace/SMART/exp Experiment outputs (logs, checkpoints)
EOF
    ;;

  *)
    # Pass-through: allow running arbitrary python commands
    echo "[SMART] Running: python $CMD $*"
    exec python "$CMD" "$@"
    ;;

esac
