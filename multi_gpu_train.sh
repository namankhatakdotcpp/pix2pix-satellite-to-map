#!/bin/bash
# -------------------------------------------------------------
# FIXED training command for IIT Mandi 8x A6000
# Root cause of SSIM=0.07: perceptual_lambda=10 + fm_lambda=10
# was crushing the generator. Fixed weights below.
# -------------------------------------------------------------

set -euo pipefail

# Make all 8 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# TF memory growth (prevents OOM on A6000 48GB each)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# XLA compilation for speed (optional, adds compile time in first epoch)
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Mixed precision for speedup on A6000 (Ampere architecture)
export TF_ENABLE_AUTO_MIXED_PRECISION=1

echo "===== FIXED RUN - IIT Mandi 8x A6000 ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python src/train.py \
  --multi_gpu \
  --mode full \
  --data_dir data/train \
  --test_dir data/test \
  --results_dir outputs/test_results_v2 \
  --savedir saved_models_v2 \
  --logdir logs_v2 \
  --split_order map_sat \
  --epochs 200 \
  --batch_size 8 \
  --gen_lr 2e-4 \
  --disc_lr 1e-4 \
  --lambda_l1 50 \
  --lambda_l1_end 50 \
  --decay_epoch 100 \
  --warmup_epochs 5 \
  --lambda_perc 5 \
  --lambda_fm 10 \
  --gan_mode lsgan \
  --label_smoothing 0.1 \
  --label_noise 0.02 \
  --disc_update_interval 1 \
  --gen_updates 2 \
  --disc_input_noise_std 0.05 \
  --require_gpu \
  --generator_norm instance \
  --res_blocks 0 \
  --cache_dataset true \
  --save_every 10 \
  --eval_every 10 \
  --sample_every 1 \
  --export

echo "===== Training Complete ====="
echo "View TensorBoard: tensorboard --logdir logs_v2"
