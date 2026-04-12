#!/bin/bash
# ---------------------------------------------------------------------
# IIT Mandi Server: 8x NVIDIA RTX A6000 Training Script
# Run: bash multi_gpu_train.sh
# ---------------------------------------------------------------------

set -euo pipefail

# Make all 8 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# TF memory growth (prevents OOM on A6000 48GB each)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# XLA compilation for speed (optional, adds compile time in first epoch)
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Mixed precision for speedup on A6000 (Ampere architecture)
export TF_ENABLE_AUTO_MIXED_PRECISION=1

echo "===== IIT Mandi 8x A6000 Training ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
else
  echo "[WARN] nvidia-smi not found"
fi
echo ""

python src/train.py \
  --multi_gpu \
  --mode full \
  --data_dir data/train \
  --test_dir data/test \
  --results_dir outputs/test_results \
  --savedir saved_models \
  --logdir logs \
  --split_order map_sat \
  --auto_split_detect \
  --require_gpu \
  --epochs 200 \
  --batch_size 8 \
  --lr 2e-4 \
  --lambda_l1 100 \
  --label_smoothing 0.05 \
  --decay_epoch 100 \
  --sample_every 5 \
  --save_every 10 \
  --eval_every 10 \
  --export

echo "===== Training Complete ====="
echo "View TensorBoard: tensorboard --logdir logs"
