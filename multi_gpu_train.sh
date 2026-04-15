#!/bin/bash
# -------------------------------------------------------------
# FIXED training command for IIT Mandi 8x A6000
# Root cause of SSIM=0.07: perceptual_lambda=10 + fm_lambda=10
# was crushing the generator. Fixed weights below.
# -------------------------------------------------------------

set -euo pipefail

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "[ERROR] python/python3 not found in PATH"
  exit 1
fi

# Robust CUDA library-path discovery
append_libdir() {
  local d="$1"
  [[ -d "$d" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$d:"*) ;;
    *) LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
}

for d in \
  /usr/local/cuda/lib64 \
  /usr/local/cuda-*/lib64 \
  /opt/cuda/lib64 \
  /opt/cuda-*/lib64 \
  /usr/lib/x86_64-linux-gnu \
  /usr/lib/wsl/lib
do
  append_libdir "$d"
done

if command -v ldconfig >/dev/null 2>&1; then
  while IFS= read -r libdir; do
    append_libdir "$libdir"
  done < <(
    ldconfig -p 2>/dev/null \
      | awk '/libcudart\.so|libcublas\.so|libcudnn\.so/ {print $NF}' \
      | xargs -r -n1 dirname \
      | sort -u
  )
fi

export LD_LIBRARY_PATH
echo "[CUDA] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"

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

GPU_COUNT="$($PYTHON_BIN - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPUs:', gpus)
print(len(gpus))
PY
)"

echo "$GPU_COUNT"
GPU_COUNT="$(echo "$GPU_COUNT" | tail -n1)"

if [[ "$GPU_COUNT" -eq 0 ]]; then
  echo "[ERROR] TensorFlow still sees 0 GPUs after CUDA path setup."
  echo "[ERROR] Check CUDA/cuDNN compatibility for your TensorFlow version on this server."
  exit 1
fi

if [[ "$GPU_COUNT" -lt 2 ]]; then
  echo "[WARN] Only $GPU_COUNT GPU visible to TensorFlow; running without --multi_gpu."
  MULTI_GPU_FLAG=()
else
  MULTI_GPU_FLAG=(--multi_gpu)
fi

"$PYTHON_BIN" src/train.py \
  "${MULTI_GPU_FLAG[@]}" \
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
echo "View TensorBoard: tensorboard --logdir logs_v3"
