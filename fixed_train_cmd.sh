#!/bin/bash
set -euo pipefail
# -------------------------------------------------------------
# FIXED training command for IIT Mandi
# Fixes: 1) No .take() on DistributedDataset
#        2) CUDA path fix
#        3) Disabled multi_gpu (use single GPU cleanly until CUDA fixed)
# -------------------------------------------------------------

# -- CUDA fix: point TF to correct library paths ----------------
# Try common locations on IIT Mandi servers
for cuda_path in /usr/local/cuda /usr/local/cuda-11 /usr/local/cuda-12 /opt/cuda; do
    if [ -d "$cuda_path/lib64" ]; then
        export LD_LIBRARY_PATH="$cuda_path/lib64:${LD_LIBRARY_PATH:-}"
        echo "[CUDA] Found at $cuda_path"
        break
    fi
done

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_AUTO_MIXED_PRECISION=1
# Suppress oneDNN noise
export TF_CPP_MIN_LOG_LEVEL=2

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "[ERROR] python/python3 not found in PATH"
  exit 1
fi

echo "===== IIT Mandi Training (FIXED) ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo ""
"$PYTHON_BIN" -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
echo ""

# NOTE: --multi_gpu is OFF until CUDA libraries are confirmed working.
# If the python check above shows GPUs, add --multi_gpu back.
"$PYTHON_BIN" src/train.py \
  --mode full \
  --data_dir data/train \
  --test_dir data/test \
  --results_dir outputs/test_results_v3 \
  --savedir saved_models_v3 \
  --logdir logs_v3 \
  --split_order map_sat \
  --epochs 200 \
  --batch_size 8 \
  --gen_lr 2e-4 \
  --disc_lr 1e-4 \
  --lambda_l1 100 \
  --lambda_l1_end 100 \
  --decay_epoch 100 \
  --warmup_epochs 5 \
  --lambda_perc 2 \
  --lambda_fm 5 \
  --gan_mode lsgan \
  --label_smoothing 0.05 \
  --label_noise 0.01 \
  --disc_update_interval 1 \
  --gen_updates 2 \
  --disc_input_noise_std 0.02 \
  --require_gpu \
  --generator_norm instance \
  --res_blocks 0 \
  --cache_dataset true \
  --save_every 10 \
  --eval_every 5 \
  --sample_every 1 \
  --export

echo "===== Training Complete ====="
echo "View TensorBoard: tensorboard --logdir logs_v2 --port 6006"
