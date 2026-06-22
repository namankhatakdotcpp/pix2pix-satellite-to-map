"""
train.py - Pix2Pix Phase 2
Phase 2 additions vs Phase 1:
  1. SelfAttention layer injected at 8x8 bottleneck of U-Net generator
  2. MS-SSIM loss (explicit structural-similarity optimisation)
  3. R1 gradient penalty on real inputs (prevents discriminator collapse)
  4. Cosine-annealing LR schedule replacing linear decay
  5. CSV training log written every epoch
  6. Best-model checkpoint keyed on SSIM
  7. analyze_log() utility printed at end of run
  8. Updated default hyperparams based on Phase 1 failure analysis
"""

import os

# TensorFlow Configuration — MUST be set BEFORE importing TensorFlow
# NOTE: CUDA_VISIBLE_DEVICES is set via shell, NOT in code
# (set via: CUDA_VISIBLE_DEVICES=2 python src/train.py)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DISABLE_LAYOUT_OPTIMIZER"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import csv
import argparse
import time
import math
import numpy as np
import tensorflow as tf
import imageio
from pathlib import Path
from tqdm import tqdm

# TensorFlow Addons removed due to version incompatibility with TF 2.19+
# SpectralNormalization disabled for stability (use LayerNormalization instead)
SpectralNormalization = None
TFA_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("[WARN] pip install scikit-image for SSIM/PSNR metrics")


# -------------------------------------------------
# DATA PIPELINE
# -------------------------------------------------

def list_image_files(data_dir):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(tf.io.gfile.glob(os.path.join(data_dir, ext)))
    return sorted(files)


def load_image(image_path, split_order="map_sat"):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    w = tf.shape(img)[1] // 2
    left = img[:, :w, :]
    right = img[:, w:, :]

    if split_order == "sat_map":
        satellite, real_map = left, right
    else:
        real_map, satellite = left, right

    real_map = tf.cast(real_map, tf.float32)
    satellite = tf.cast(satellite, tf.float32)
    return satellite, real_map


def normalize(satellite, real_map):
    return (satellite / 127.5) - 1.0, (real_map / 127.5) - 1.0


def random_jitter(satellite, real_map):
    satellite = tf.image.resize(satellite, [286, 286], method='nearest')
    real_map = tf.image.resize(real_map, [286, 286], method='nearest')

    stacked = tf.stack([satellite, real_map], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, 256, 256, 3])
    satellite, real_map = cropped[0], cropped[1]

    if tf.random.uniform(()) > 0.5:
        satellite = tf.image.flip_left_right(satellite)
        real_map = tf.image.flip_left_right(real_map)

    return satellite, real_map


def load_train_image(image_path, split_order="map_sat"):
    satellite, real_map = load_image(image_path, split_order)
    satellite, real_map = random_jitter(satellite, real_map)
    return normalize(satellite, real_map)


def load_test_image(image_path, split_order="map_sat"):
    satellite, real_map = load_image(image_path, split_order)
    satellite = tf.image.resize(satellite, [256, 256])
    real_map = tf.image.resize(real_map, [256, 256])
    return normalize(satellite, real_map)


def build_dataset(data_dir, batch_size, is_train=True, split_order="map_sat",
                  use_cache=True, buffer_size=400):
    """
    SIMPLIFIED DATASET PIPELINE FOR DEBUGGING
    Removed: cache, prefetch, AUTOTUNE, map parallelism
    This isolates tf.data variant tensor issues.
    """
    image_files = list_image_files(data_dir)
    if not image_files:
        raise RuntimeError(f"No images found in {data_dir}")

    ds = tf.data.Dataset.from_tensor_slices(image_files)
    
    if is_train:
        ds = ds.shuffle(len(image_files), reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p: load_train_image(p, split_order),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(
            lambda p: load_test_image(p, split_order),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(1)
        ds = ds.prefetch(tf.data.AUTOTUNE)

    # Force CPU dataset execution - disable automatic sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    ds = ds.with_options(options)

    return ds


# -------------------------------------------------
# MODEL ARCHITECTURE
# -------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        c = input_shape[-1]
        self.scale = self.add_weight(
            name='scale',
            shape=(c,),
            initializer='ones',
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(c,),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    """
    Channel self-attention (SAGAN-style) for the generator bottleneck.
    Learns long-range spatial dependencies — critical for road-network coherence.
    gamma is initialised to 0 so the layer starts as an identity and
    gradually learns to contribute attention.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        reduced = max(1, channels // 8)
        self.q_conv = tf.keras.layers.Conv2D(reduced, 1, use_bias=False)
        self.k_conv = tf.keras.layers.Conv2D(reduced, 1, use_bias=False)
        self.v_conv = tf.keras.layers.Conv2D(channels, 1, use_bias=False)
        self.out_conv = tf.keras.layers.Conv2D(channels, 1, use_bias=False)
        self._channels = channels
        self._reduced = reduced

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # Save original dtype for mixed precision compatibility
        orig_dtype = x.dtype
        
        shape = tf.shape(x)
        B, H, W = shape[0], shape[1], shape[2]
        HW = H * W
        reduced = self._reduced
        C = self._channels

        q = tf.reshape(self.q_conv(x), [B, HW, reduced])
        k = tf.reshape(self.k_conv(x), [B, HW, reduced])
        v = tf.reshape(self.v_conv(x), [B, HW, C])

        # Cast to float32 for numerically stable attention computation
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        v = tf.cast(v, tf.float32)

        scale = tf.math.sqrt(tf.cast(reduced, tf.float32))
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale, axis=-1)

        attended = tf.reshape(tf.matmul(attn, v), [B, H, W, C])
        
        # Cast output back to original dtype (FP16 if mixed precision, else FP32)
        attended = tf.cast(attended, orig_dtype)
        
        return self.gamma * self.out_conv(attended) + x

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self._channels})
        return config


@tf.keras.utils.register_keras_serializable()
class NoDropout(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        return x

    def get_config(self):
        return super().get_config()


def _norm(norm_type):
    if norm_type == 'instance':
        return InstanceNormalization()
    return tf.keras.layers.BatchNormalization()


def downsample(filters, size, apply_norm=True, norm_type='instance'):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=init, use_bias=False,
    ))
    if apply_norm:
        block.add(_norm(norm_type))
    block.add(tf.keras.layers.LeakyReLU())
    return block


def upsample(filters, size, apply_dropout=False, norm_type='instance'):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=init, use_bias=False,
    ))
    block.add(_norm(norm_type))
    if apply_dropout:
        block.add(NoDropout())
    block.add(tf.keras.layers.ReLU())
    return block


def build_generator(norm_type='instance'):
    """
    U-Net generator with a SelfAttention module inserted at the 8x8 bottleneck
    (after the 5th downsampling layer).  Last layer is forced float32 for tanh
    stability under mixed precision.
    """
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    init = tf.random_normal_initializer(0., 0.02)

    down_stack = [
        downsample(64,  4, apply_norm=False, norm_type=norm_type),  # → 128
        downsample(128, 4, norm_type=norm_type),                     # → 64
        downsample(256, 4, norm_type=norm_type),                     # → 32
        downsample(512, 4, norm_type=norm_type),                     # → 16
        downsample(512, 4, norm_type=norm_type),                     # → 8
        downsample(512, 4, norm_type=norm_type),                     # → 4
        downsample(512, 4, norm_type=norm_type),                     # → 2
        downsample(512, 4, norm_type=norm_type),                     # → 1 (bottleneck)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, norm_type=norm_type),
        upsample(256, 4, norm_type=norm_type),
        upsample(128, 4, norm_type=norm_type),
        upsample(64,  4, norm_type=norm_type),
    ]

    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=init, activation='tanh', dtype='float32',
    )

    x = inputs
    skips = []
    for i, down in enumerate(down_stack):
        x = down(x)
        # Inject self-attention at 8x8 spatial resolution (index 4, after 5th downsample).
        if i == 4:
            x = SelfAttention(512, name='bottleneck_attn')(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    return tf.keras.Model(inputs=inputs, outputs=last(x))


def residual_block(x, filters):
    """Standard residual block: conv-norm-relu-conv-norm + skip connection."""
    skip = x
    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    x = InstanceNormalization()(x)
    return tf.keras.layers.Add()([x, skip])


def build_global_generator(norm_type='instance'):
    """
    G1: U-Net generator at half resolution (128x128).
    Identical architecture to build_generator() but with one fewer
    downsample/upsample pair to match the smaller spatial dimensions.
    SelfAttention at 4x4 bottleneck (index 3 after 4th downsample).
    """
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    init = tf.random_normal_initializer(0., 0.02)

    down_stack = [
        downsample(64,  4, apply_norm=False, norm_type=norm_type),  # → 64
        downsample(128, 4, norm_type=norm_type),                     # → 32
        downsample(256, 4, norm_type=norm_type),                     # → 16
        downsample(512, 4, norm_type=norm_type),                     # → 8
        downsample(512, 4, norm_type=norm_type),                     # → 4
        downsample(512, 4, norm_type=norm_type),                     # → 2
        downsample(512, 4, norm_type=norm_type),                     # → 1 (bottleneck)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, norm_type=norm_type),
        upsample(256, 4, norm_type=norm_type),
        upsample(128, 4, norm_type=norm_type),
        upsample(64,  4, norm_type=norm_type),
    ]

    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=init, activation='tanh', dtype='float32',
    )

    x = inputs
    skips = []
    for i, down in enumerate(down_stack):
        x = down(x)
        if i == 3:
            x = SelfAttention(512, name='g1_bottleneck_attn')(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    return tf.keras.Model(inputs=inputs, outputs=last(x), name='global_generator')


def build_local_enhancer(global_generator, input_shape=(256, 256, 3)):
    """
    G2: takes 256x256 input, downsamples to 128x128 for G1,
    combines G1 features with local front-end, refines via ResBlocks.
    """
    init = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=input_shape)

    # Front-end: downsample 256 -> 128 with feature extraction
    front = tf.keras.layers.Conv2D(64, 7, strides=1, padding='same',
                                    kernel_initializer=init)(inp)
    front = InstanceNormalization()(front)
    front = tf.keras.layers.ReLU()(front)

    front = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                                    kernel_initializer=init)(front)
    front = InstanceNormalization()(front)
    front = tf.keras.layers.ReLU()(front)  # 128x128x64

    # Downsample input for G1
    inp_downsampled = tf.keras.layers.AveragePooling2D(pool_size=2)(inp)
    global_out = global_generator(inp_downsampled)  # 128x128x3

    # Combine G1 output with local front-end features
    combined = tf.keras.layers.Concatenate()([front, global_out])  # 128x128x67

    # Back-end: upsample to 256, refine with residual blocks
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same',
                                         kernel_initializer=init)(combined)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)  # 256x256x64

    for _ in range(3):
        x = residual_block(x, 64)

    x = tf.keras.layers.Conv2D(3, 7, padding='same', activation='tanh',
                                kernel_initializer=init, dtype='float32')(x)

    return tf.keras.Model(inputs=inp, outputs=x, name='local_enhancer')


def build_discriminator(use_spectral_norm=False, input_size=256):
    """
    PatchGAN discriminator with LayerNorm.
    Returns [patch_logits, f1, f2, f3, f4] for feature-matching loss.
    input_size: spatial resolution of inputs (256 for full-res, 128 for half, 64 for quarter).
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[input_size, input_size, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[input_size, input_size, 3], name='target_image')
    x = tf.keras.layers.Concatenate()([inp, tar])

    def conv2d(*args, **kwargs):
        layer = tf.keras.layers.Conv2D(*args, **kwargs)
        if use_spectral_norm:
            return SpectralNormalization(layer)
        return layer

    def disc_downsample(x_in, filters, size, apply_norm=True):
        y = conv2d(filters, size, strides=2, padding='same',
                   kernel_initializer=initializer, use_bias=not apply_norm)(x_in)
        if apply_norm:
            y = tf.keras.layers.LayerNormalization()(y)
        return tf.keras.layers.LeakyReLU(0.2)(y)

    f1 = disc_downsample(x,  64,  4, apply_norm=False)
    f2 = disc_downsample(f1, 128, 4)
    f3 = disc_downsample(f2, 256, 4)

    y = conv2d(512, 4, strides=1, padding='same',
               kernel_initializer=initializer, use_bias=False)(f3)
    y = tf.keras.layers.LayerNormalization()(y)
    f4 = tf.keras.layers.LeakyReLU(0.2)(y)

    patch = conv2d(1, 4, strides=1, padding='same',
                   kernel_initializer=initializer, dtype='float32')(f4)

    return tf.keras.Model(inputs=[inp, tar], outputs=[patch, f1, f2, f3, f4])


def build_multiscale_discriminators(use_spectral_norm=False):
    """Returns 3 PatchGAN discriminators for full-res, half-res, quarter-res inputs."""
    d1 = build_discriminator(use_spectral_norm=use_spectral_norm, input_size=256)
    d2 = build_discriminator(use_spectral_norm=use_spectral_norm, input_size=128)
    d3 = build_discriminator(use_spectral_norm=use_spectral_norm, input_size=64)
    return [d1, d2, d3]


# -------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------

bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_fake_patch, gen_output, target, lambda_l1=100.0, gan_mode='lsgan'):
    if gan_mode == 'lsgan':
        adv = tf.reduce_mean(tf.square(tf.cast(disc_fake_patch, tf.float32) - 1.0))
    else:
        adv = bce_loss(tf.ones_like(disc_fake_patch), disc_fake_patch)

    l1 = tf.reduce_mean(tf.abs(tf.cast(target, tf.float32) - tf.cast(gen_output, tf.float32)))
    return adv + lambda_l1 * l1, adv, l1


def discriminator_loss(disc_real_patch, disc_fake_patch,
                       label_smoothing=0.1, label_noise=0.02, gan_mode='lsgan'):
    real_p = tf.cast(disc_real_patch, tf.float32)
    fake_p = tf.cast(disc_fake_patch, tf.float32)

    if gan_mode == 'lsgan':
        # Soft real target: 1 - label_smoothing (default 0.85).
        real_target = 1.0 - label_smoothing
        real_loss = tf.reduce_mean(tf.square(real_p - real_target))
        fake_loss = tf.reduce_mean(tf.square(fake_p))
        return 0.5 * (real_loss + fake_loss)

    real_label = tf.ones_like(real_p) * (1.0 - label_smoothing)
    fake_label = tf.zeros_like(fake_p) + label_smoothing
    if label_noise > 0:
        real_label = tf.clip_by_value(
            real_label + tf.random.uniform(tf.shape(real_label), -label_noise, label_noise),
            0.0, 1.0,
        )
        fake_label = tf.clip_by_value(
            fake_label + tf.random.uniform(tf.shape(fake_label), -label_noise, label_noise),
            0.0, 1.0,
        )
    return bce_loss(real_label, real_p) + bce_loss(fake_label, fake_p)


def ms_ssim_loss_fn(real, fake):
    """
    Multi-Scale SSIM loss.  Returns 1 - MS-SSIM so that minimising it
    maximises structural similarity across 5 spatial scales.
    Images must be in [-1, 1]; shifted to [0, 2] for the TF call.
    """
    real_f = tf.cast(real, tf.float32) + 1.0
    fake_f = tf.cast(fake, tf.float32) + 1.0
    ms_val = tf.image.ssim_multiscale(real_f, fake_f, max_val=2.0)
    return 1.0 - tf.reduce_mean(ms_val)


def perceptual_loss_fn(perc_model, real_img, fake_img):
    if perc_model is None:
        return tf.constant(0.0)

    from tensorflow.keras.applications.vgg19 import preprocess_input

    def prep(x):
        x = tf.cast(x, tf.float32)
        x = (x + 1.0) * 127.5
        x = tf.image.resize(x, [224, 224])
        return preprocess_input(x)

    real_f = perc_model(prep(real_img), training=False)
    fake_f = perc_model(prep(fake_img), training=False)

    if not isinstance(real_f, (list, tuple)):
        real_f, fake_f = [real_f], [fake_f]

    losses = [tf.reduce_mean(tf.abs(tf.stop_gradient(r) - f)) for r, f in zip(real_f, fake_f)]
    raw_loss = tf.add_n(losses) / float(len(losses))
    return raw_loss / 128.0  # normalize VGG19 range ~110-160 → ~0.85-1.25


def feature_matching_loss_fn(real_feats, fake_feats):
    losses = [
        tf.reduce_mean(tf.abs(tf.stop_gradient(tf.cast(r, tf.float32)) - tf.cast(f, tf.float32)))
        for r, f in zip(real_feats, fake_feats)
    ]
    return tf.add_n(losses) / float(len(losses))


# -------------------------------------------------
# LR SCHEDULE
# -------------------------------------------------

def cosine_lr(initial_lr, epoch, total_epochs, min_lr=1e-7, warmup_epochs=0):
    """Cosine annealing with linear warm-up."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return initial_lr * ((epoch + 1) / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return float(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))


# -------------------------------------------------
# TRAIN STEP
# -------------------------------------------------

def get_pyramid(img, num_scales):
    """Returns [full_res, half_res, quarter_res, ...] via average pooling."""
    pyramid = [img]
    x = img
    for _ in range(num_scales - 1):
        x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID')
        pyramid.append(x)
    return pyramid


@tf.function
def train_step(satellite, real_map, generator, discriminators,
               gen_opt, disc_opt, lambda_l1, label_smoothing, label_noise,
               fm_lambda, perc_model, perc_lambda, gan_mode, update_disc,
               gen_updates, disc_input_noise_std, ms_ssim_lambda, r1_gamma):
    """
    Single distributed training step with multi-scale discriminator support.

    discriminators: list of PatchGAN discriminators (len 1 for vanilla, len 3 for Pix2PixHD).
    Generator loss  = LSGAN_adv + λ_l1*L1 + λ_ms*MS-SSIM + λ_fm*FM + λ_perc*Perceptual
    Discriminator loss = LSGAN + R1 gradient penalty (prevents collapse)
    """
    _ZEROS = tf.constant(0.0, dtype=tf.float32)
    _ZERO_RETURN = (_ZEROS, _ZEROS, _ZEROS, _ZEROS, _ZEROS, _ZEROS, _ZEROS, _ZEROS)
    num_scales = len(discriminators)

    def add_noise(x):
        if disc_input_noise_std <= 0:
            return x
        return x + tf.random.normal(tf.shape(x), 0.0, disc_input_noise_std, dtype=x.dtype)

    gen_total = _ZEROS
    gen_adv   = _ZEROS
    gen_l1    = _ZEROS
    fm        = _ZEROS
    perc      = _ZEROS
    ms_ssim   = _ZEROS
    gen_map   = generator(satellite, training=False)

    # Collect all discriminator trainable variables for gradient computation
    disc_all_vars = []
    for d in discriminators:
        disc_all_vars.extend(d.trainable_variables)

    # ---- Generator updates ----
    for _ in range(max(1, gen_updates)):
        with tf.GradientTape() as gt:
            gen_map = generator(satellite, training=True)

            real_map_f32_gen = tf.cast(real_map, tf.float32)
            gen_map_f32 = tf.cast(gen_map, tf.float32)

            # Build pyramids for multi-scale discriminators
            sat_pyramid  = get_pyramid(tf.cast(satellite, tf.float32), num_scales)
            real_pyramid = get_pyramid(real_map_f32_gen, num_scales)
            fake_pyramid = get_pyramid(gen_map_f32, num_scales)

            # Accumulate adversarial + FM loss across all discriminator scales
            adv_sum = _ZEROS
            fm_sum  = _ZEROS
            for d, sat_s, real_s, fake_s in zip(discriminators, sat_pyramid, real_pyramid, fake_pyramid):
                d_real_out = d([sat_s, add_noise(real_s)], training=True)
                d_fake_out = d([sat_s, add_noise(fake_s)], training=True)

                if gan_mode == 'lsgan':
                    adv_s = tf.reduce_mean(tf.square(tf.cast(d_fake_out[0], tf.float32) - 1.0))
                else:
                    adv_s = bce_loss(tf.ones_like(d_fake_out[0]), d_fake_out[0])
                adv_sum += adv_s

                if fm_lambda > 0:
                    fm_s = feature_matching_loss_fn(d_real_out[1:], d_fake_out[1:])
                    fm_sum += fm_s

            gen_adv = adv_sum / tf.cast(num_scales, tf.float32)
            fm = fm_sum / tf.cast(num_scales, tf.float32) if fm_lambda > 0 else _ZEROS

            l1 = tf.reduce_mean(tf.abs(tf.cast(real_map, tf.float32) - gen_map_f32))
            gen_l1 = l1

            perc = perceptual_loss_fn(perc_model, real_map, gen_map)

            if ms_ssim_lambda > 0:
                ms_ssim = ms_ssim_loss_fn(real_map, gen_map)
            else:
                ms_ssim = _ZEROS

            gen_adv = tf.cast(gen_adv, tf.float32)
            gen_l1 = tf.cast(gen_l1, tf.float32)
            fm = tf.cast(fm, tf.float32)
            perc = tf.cast(perc, tf.float32)
            ms_ssim = tf.cast(ms_ssim, tf.float32)

            gen_adv = tf.clip_by_value(gen_adv, 0.0, 50.0)
            perc    = tf.clip_by_value(perc,    0.0, 200.0)

            gen_total = (
                gen_adv
                + tf.cast(lambda_l1, tf.float32) * gen_l1
                + tf.cast(fm_lambda, tf.float32) * fm
                + tf.cast(perc_lambda, tf.float32) * perc
                + tf.cast(ms_ssim_lambda, tf.float32) * ms_ssim
            )

            gen_total = tf.clip_by_value(gen_total, 0.0, 500.0)

            if not tf.math.is_finite(gen_total):
                tf.print("[WARN] gen_total non-finite, skipping batch")
                return _ZERO_RETURN

        gen_grads = gt.gradient(gen_total, generator.trainable_variables)
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, 1.0)
        gen_grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            if g is not None else None
            for g in gen_grads
        ]
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))

    # ---- Discriminator update with R1 gradient penalty (multi-scale) ----
    real_map_f32 = tf.cast(real_map, tf.float32)
    gen_map_disc = tf.stop_gradient(tf.cast(gen_map, tf.float32))

    sat_pyramid  = get_pyramid(tf.cast(satellite, tf.float32), num_scales)
    real_pyramid = get_pyramid(real_map_f32, num_scales)
    fake_pyramid = get_pyramid(gen_map_disc, num_scales)

    with tf.GradientTape() as dt:
        disc_loss_sum = _ZEROS
        r1_penalty_sum = _ZEROS

        for d, sat_s, real_s, fake_s in zip(discriminators, sat_pyramid, real_pyramid, fake_pyramid):
            # R1 penalty on this scale's real input
            with tf.GradientTape() as r1_tape:
                r1_tape.watch(real_s)
                d_real_out = d([sat_s, add_noise(real_s)], training=True)
                real_patch = tf.cast(d_real_out[0], tf.float32)
            r1_grads = r1_tape.gradient(real_patch, real_s)
            r1_pen = r1_gamma * 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(r1_grads), axis=[1, 2, 3])
            )
            r1_penalty_sum += r1_pen

            d_fake_out = d([sat_s, add_noise(fake_s)], training=True)
            d_loss = discriminator_loss(
                d_real_out[0], d_fake_out[0],
                label_smoothing, label_noise, gan_mode,
            )
            disc_loss_sum += tf.cast(d_loss, tf.float32)

        disc_loss_avg = disc_loss_sum / tf.cast(num_scales, tf.float32)
        r1_penalty = r1_penalty_sum / tf.cast(num_scales, tf.float32)

        disc_loss_avg = tf.clip_by_value(disc_loss_avg, 0.0, 50.0)
        r1_penalty = tf.clip_by_value(r1_penalty, 0.0, 100.0)

        disc_total = disc_loss_avg + r1_penalty
        disc_total = tf.clip_by_value(disc_total, 0.0, 150.0)

        if not tf.math.is_finite(disc_total):
            tf.print("[WARN] disc_total non-finite, skipping batch")
            return _ZERO_RETURN

    disc_grads = dt.gradient(disc_total, disc_all_vars)
    disc_grads, _ = tf.clip_by_global_norm(disc_grads, 1.0)
    disc_grads = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        if g is not None else None
        for g in disc_grads
    ]
    if update_disc:
        disc_opt.apply_gradients(zip(disc_grads, disc_all_vars))

    return gen_total, gen_adv, gen_l1, disc_total, fm, perc, ms_ssim, r1_penalty


# -------------------------------------------------
# EVALUATION & VISUALISATION
# -------------------------------------------------

def evaluate_metrics(generator, test_ds, num_samples=50):
    if not METRICS_AVAILABLE:
        return {}

    maes, ssims, psnrs = [], [], []
    for sat, real_map in test_ds.take(num_samples):
        gen_map = generator(sat, training=False)
        real = ((real_map[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        pred = ((gen_map[0].numpy()  + 1) * 127.5).clip(0, 255).astype(np.uint8)
        maes.append(np.mean(np.abs(real.astype(float) - pred.astype(float))) / 255.0)
        ssims.append(ssim_fn(real, pred, channel_axis=2, data_range=255))
        psnrs.append(psnr_fn(real, pred, data_range=255))

    return {
        "MAE":  float(np.mean(maes)),
        "SSIM": float(np.mean(ssims)),
        "PSNR": float(np.mean(psnrs)),
    }


def save_sample(generator, test_ds, results_dir, epoch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for sat, real_map in test_ds.take(1):
        gen_map = generator(sat, training=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(
            axes,
            [sat[0], gen_map[0], real_map[0]],
            ['Satellite Input', 'Generated Map', 'Ground Truth'],
        ):
            ax.imshow((img.numpy() * 0.5 + 0.5).clip(0, 1))
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        plt.suptitle(f'Epoch {epoch:03d}', fontsize=14)
        plt.tight_layout()
        path = os.path.join(results_dir, f'epoch_{epoch:03d}.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        return path


def make_gif(results_dir, output_path):
    """
    Create GIF from epoch preview images with robust normalization.
    Handles inconsistent image sizes, formats, and channel counts.
    """
    from PIL import Image
    
    # Find all preview images
    image_files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    
    if not image_files:
        print(f"[GIF] No images found in {results_dir}")
        return
    
    frames = []
    TARGET_SIZE = (256, 256)  # Consistent frame size
    
    for file in image_files:
        try:
            img = Image.open(file)
            
            # Convert EVERYTHING to RGB (handles RGBA, grayscale, etc.)
            img = img.convert("RGB")
            
            # Force identical size
            img = img.resize(TARGET_SIZE)
            
            frame = np.array(img)
            
            # Debug: print shape for problematic frames
            print(f"[GIF] {os.path.basename(file)}: {frame.shape}")
            
            frames.append(frame)
        
        except Exception as e:
            print(f"[GIF] Skipping {file}: {e}")
    
    if len(frames) == 0:
        print("[GIF] No valid frames found.")
        return
    
    # All frames now guaranteed to be (256, 256, 3) RGB uint8
    imageio.mimsave(output_path, frames, fps=4)
    print(f"[GIF] Saved: {output_path} ({len(frames)} frames)")


# -------------------------------------------------
# CSV LOGGING
# -------------------------------------------------

_CSV_HEADER = [
    'epoch', 'g_loss', 'd_loss', 'l1', 'adv', 'fm', 'perc',
    'ms_ssim_loss', 'r1', 'mae', 'ssim', 'psnr', 'gen_lr', 'disc_lr',
]


def init_csv_log(logdir):
    path = os.path.join(logdir, 'training_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(_CSV_HEADER)
    return path


def append_csv_log(path, row: dict):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([row.get(k, '') for k in _CSV_HEADER])


def analyze_log(log_path):
    """Print the epoch with the best SSIM, PSNR and MAE from training_log.csv."""
    if not os.path.exists(log_path):
        return
    with open(log_path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('ssim')]

    if not rows:
        print("[LOG] No metric rows in log yet.")
        return

    best_ssim_row = max(rows, key=lambda r: float(r['ssim']))
    best_psnr_row = max(rows, key=lambda r: float(r['psnr']))
    best_mae_row  = min(rows, key=lambda r: float(r['mae']))

    print("\n" + "=" * 55)
    print("  TRAINING LOG ANALYSIS")
    print("=" * 55)
    print(f"  Best SSIM : Epoch {best_ssim_row['epoch']:>4s} | "
          f"SSIM={float(best_ssim_row['ssim']):.4f} | "
          f"PSNR={float(best_ssim_row['psnr']):.2f}dB | "
          f"MAE={float(best_ssim_row['mae']):.4f}")
    print(f"  Best PSNR : Epoch {best_psnr_row['epoch']:>4s} | "
          f"PSNR={float(best_psnr_row['psnr']):.2f}dB | "
          f"SSIM={float(best_psnr_row['ssim']):.4f}")
    print(f"  Best MAE  : Epoch {best_mae_row['epoch']:>4s} | "
          f"MAE={float(best_mae_row['mae']):.4f} | "
          f"SSIM={float(best_mae_row['ssim']):.4f}")
    print("=" * 55 + "\n")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError("Expected boolean value")


def main():
    ap = argparse.ArgumentParser(description="Pix2Pix Phase 2 — Satellite→Map")

    # Paths
    ap.add_argument('--data_dir',    default='data/train')
    ap.add_argument('--test_dir',    default='data/test')
    ap.add_argument('--results_dir', default='outputs/test_results')
    ap.add_argument('--savedir',     default='saved_models')
    ap.add_argument('--logdir',      default='logs')

    # Training schedule
    ap.add_argument('--epochs',         type=int,   default=300)
    ap.add_argument('--batch_size',     type=int,   default=2)
    ap.add_argument('--warmup_epochs',  type=int,   default=10)
    ap.add_argument('--save_every',     type=int,   default=10)
    ap.add_argument('--eval_every',     type=int,   default=5)
    ap.add_argument('--sample_every',   type=int,   default=1)
    ap.add_argument('--lr_schedule',    default='linear', choices=['cosine', 'linear'])

    # Learning rates
    ap.add_argument('--lr',      type=float, default=None,
                    help='Legacy flag: overrides both gen_lr and disc_lr.')
    ap.add_argument('--gen_lr',  type=float, default=1e-4)
    ap.add_argument('--disc_lr', type=float, default=5e-5)
    ap.add_argument('--min_lr',  type=float, default=1e-7,
                    help='Floor LR for cosine annealing.')

    # Loss weights
    ap.add_argument('--lambda_l1',     type=float, default=100.0,
                    help='L1 pixel loss weight at epoch 0.')
    ap.add_argument('--lambda_l1_end', type=float, default=100.0,
                    help='L1 pixel loss weight at final epoch (linear decay).')
    ap.add_argument('--ms_ssim_lambda',             type=float, default=0.0,
                    help='MS-SSIM loss weight (explicit SSIM optimisation).')
    ap.add_argument('--perceptual_lambda', '--lambda_perc',
                    dest='perceptual_lambda',       type=float, default=10.0)
    ap.add_argument('--feature_matching_lambda', '--lambda_fm',
                    dest='feature_matching_lambda', type=float, default=20.0)
    ap.add_argument('--r1_gamma',      type=float, default=1.0,
                    help='R1 gradient-penalty coefficient (0 = disabled).')

    # GAN stability knobs
    ap.add_argument('--gan_mode',           default='lsgan', choices=['lsgan', 'bce'])
    ap.add_argument('--label_smoothing',    type=float, default=0.15)
    ap.add_argument('--label_noise',        type=float, default=0.02)
    ap.add_argument('--disc_update_interval', type=int, default=1)
    ap.add_argument('--gen_updates',        type=int,   default=1,
                    help='Generator updates per discriminator update (TTUR).')
    ap.add_argument('--disc_input_noise_std', type=float, default=0.1,
                    help='Disc input noise std, annealed to 0 over first 50%% of training.')

    # Architecture
    ap.add_argument('--generator_norm', default='instance', choices=['instance', 'batch'])
    ap.add_argument('--spectral_norm',  action='store_true')
    ap.add_argument('--use_pix2pixhd', action='store_true', default=False,
                    help='Use Pix2PixHD architecture (coarse-to-fine generator + multi-scale discriminator)')
    ap.add_argument('--g1_pretrain_epochs', type=int, default=0,
                    help='Epochs to train G1 alone before introducing G2 (0 = train jointly)')

    # Dataset
    ap.add_argument('--split_order',   default='map_sat', choices=['map_sat', 'sat_map'])
    ap.add_argument('--cache_dataset', nargs='?', const=True, default=False, type=str2bool)

    # Runtime
    ap.add_argument('--mode',       default='full', choices=['full', 'demo'])
    ap.add_argument('--demo_steps', type=int, default=5)
    ap.add_argument('--restore',    action='store_true', help='Resume from latest checkpoint.')
    ap.add_argument('--resume',     type=str, default=None, help='Path to saved generator checkpoint (e.g., saved_models/generator_epoch_050.keras)')
    ap.add_argument('--resume_disc', default=None,
                    help='Path to discriminator checkpoint to resume from')
    ap.add_argument('--export',     action='store_true')
    ap.add_argument('--multi_gpu',  action='store_true')
    ap.add_argument('--require_gpu',action='store_true')
    ap.add_argument('--mixed_precision', action='store_true', default=False)

    args = ap.parse_args()

    if args.lr is not None:
        args.gen_lr = float(args.lr)
        args.disc_lr = float(args.lr)

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[AMP] mixed_float16 enabled")

    # GPU / strategy
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[GPU] Detected: {len(gpus)} GPU(s)")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    print(f"[GPU] CUDA_VISIBLE_DEVICES={visible_devices}")
    
    # Enable memory growth on visible GPUs
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[GPU] Memory growth error: {e}")
    
    if not gpus:
        print("[WARN] No GPUs found — training will be slow.")
        if args.require_gpu:
            raise RuntimeError("--require_gpu set but no GPU visible to TensorFlow.")

    # Strategy selection based on GPU availability (NOT on --multi_gpu flag)
    # This ensures true single-GPU mode when only 1 GPU is visible
    if len(gpus) > 1:
        print(f"[GPU] Using MirroredStrategy on {len(gpus)} GPUs")
        strategy = tf.distribute.MirroredStrategy()
        effective_batch = args.batch_size * strategy.num_replicas_in_sync
    else:
        print("[GPU] Using single GPU/default strategy (no replica graph overhead)")
        strategy = tf.distribute.get_strategy()
        effective_batch = args.batch_size

    print(f"[BATCH] effective_batch={effective_batch}")

    for d in [args.results_dir, args.savedir, args.logdir]:
        os.makedirs(d, exist_ok=True)

    train_ds = build_dataset(
        args.data_dir, effective_batch, is_train=True,
        split_order=args.split_order, use_cache=args.cache_dataset,
    )
    test_ds = build_dataset(
        args.test_dir, 1, is_train=False,
        split_order=args.split_order, use_cache=args.cache_dataset,
    )

    n_train = len(list_image_files(args.data_dir))
    steps_per_epoch = max(1, int(math.ceil(n_train / effective_batch)))
    print(f"[DATA] {n_train} images | {steps_per_epoch} steps/epoch")

    with strategy.scope():
        custom_objects = {
            'InstanceNormalization': InstanceNormalization,
            'SelfAttention': SelfAttention,
            'NoDropout': NoDropout,
        }

        if args.use_pix2pixhd:
            print("[MODEL] Using Pix2PixHD architecture (multi-scale)")
            global_gen = build_global_generator(norm_type=args.generator_norm)
            generator = build_local_enhancer(global_gen)
            discriminators = build_multiscale_discriminators(use_spectral_norm=False)
        else:
            print("[MODEL] Using standard Pix2Pix architecture")
            generator = build_generator(norm_type=args.generator_norm)
            discriminators = [build_discriminator(use_spectral_norm=False)]

        # Load generator from checkpoint if --resume specified
        if args.resume is not None:
            if os.path.exists(args.resume):
                print(f"[RESUME] Loading generator weights from: {args.resume}")
                generator = tf.keras.models.load_model(
                    args.resume, custom_objects=custom_objects, compile=False)
            else:
                print(f"[WARN] Resume checkpoint not found: {args.resume}")

        # Load discriminator(s) from checkpoint if --resume_disc specified
        if args.resume_disc and os.path.exists(args.resume_disc):
            if args.use_pix2pixhd:
                loaded_discs = []
                for i in range(3):
                    path = args.resume_disc.replace('.keras', f'_d{i+1}.keras')
                    if os.path.exists(path):
                        try:
                            loaded_discs.append(tf.keras.models.load_model(path, compile=False))
                            print(f"[RESUME] Loaded discriminator d{i+1} from: {path}")
                        except Exception as e:
                            print(f"[RESUME] Could not load d{i+1}: {e}, using fresh")
                            loaded_discs.append(discriminators[i])
                    else:
                        print(f"[RESUME] d{i+1} not found at {path}, using fresh")
                        loaded_discs.append(discriminators[i])
                discriminators = loaded_discs
            else:
                try:
                    discriminators = [tf.keras.models.load_model(args.resume_disc, compile=False)]
                    print(f"[RESUME] Loaded discriminator weights from: {args.resume_disc}")
                except Exception as e:
                    print(f"[RESUME] Could not load discriminator weights: {e}")

        # SpectralNormalization removed - TensorFlow Addons incompatible with TF 2.19+
        # Using LayerNormalization in discriminator instead
        if args.spectral_norm:
            print("[INFO] SpectralNormalization disabled (TensorFlow Addons removed for TF 2.19+ compatibility)")
            args.spectral_norm = False

        perc_model = None
        if args.perceptual_lambda > 0:
            try:
                from tensorflow.keras.applications import VGG19
                base = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
                base.trainable = False
                # 4 layers: low-level textures (block1-2) + mid/high semantics (block3-4)
                outs = [base.get_layer(n).output
                        for n in ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3')]
                perc_model = tf.keras.Model(base.input, outs)
                perc_model.trainable = False
                print("[LOSS] VGG19 perceptual loss enabled (4 layers)")
            except Exception as e:
                print(f"[WARN] VGG19 unavailable ({e}), perceptual_lambda=0")
                args.perceptual_lambda = 0.0

        gen_opt  = tf.keras.optimizers.Adam(learning_rate=args.gen_lr,  beta_1=0.5, clipnorm=1.0)
        disc_opt = tf.keras.optimizers.Adam(learning_rate=args.disc_lr, beta_1=0.5, clipnorm=1.0)

        epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64)
        ckpt_kwargs = {
            'generator': generator,
            'gen_opt': gen_opt, 'disc_opt': disc_opt, 'epoch': epoch_var,
        }
        for i, d in enumerate(discriminators):
            ckpt_kwargs[f'discriminator_{i}'] = d
        ckpt = tf.train.Checkpoint(**ckpt_kwargs)

    mgr = tf.train.CheckpointManager(ckpt, args.savedir, max_to_keep=5)

    start_epoch = 0
    if args.restore and mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint)
        start_epoch = int(epoch_var.numpy())
        print(f"[CKPT] Restored from {mgr.latest_checkpoint} -> resuming from epoch {start_epoch + 1}")
    elif args.resume is not None:
        # Try to extract epoch number from resume path (e.g., generator_epoch_050.keras)
        try:
            start_epoch = int(
                os.path.basename(args.resume)
                .split("_epoch_")[1]
                .split(".")[0]
            )
            print(f"[RESUME] Extracted epoch {start_epoch} from checkpoint name")
        except:
            start_epoch = 0
            print(f"[RESUME] Could not extract epoch from checkpoint name, starting from 0")

    sw       = tf.summary.create_file_writer(args.logdir)
    csv_path = init_csv_log(args.logdir)
    best_ssim = 0.0

    arch_name = 'Pix2PixHD' if args.use_pix2pixhd else 'Pix2Pix'
    print(f"\n{'=' * 60}")
    print(f"  {arch_name} | {args.epochs} epochs | LR={args.lr_schedule}")
    print(f"  discriminators={len(discriminators)} | L1={args.lambda_l1}→{args.lambda_l1_end} | MS-SSIM={args.ms_ssim_lambda}")
    print(f"  perc={args.perceptual_lambda} | fm={args.feature_matching_lambda} | R1γ={args.r1_gamma}")
    print(f"  gen_lr={args.gen_lr} | disc_lr={args.disc_lr} | G:D={max(1,args.gen_updates)}:1")
    print(f"  gan_mode={args.gan_mode} | norm={args.generator_norm} | spectral={args.spectral_norm}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # LR schedule
        if args.lr_schedule == 'cosine':
            gen_lr_now  = cosine_lr(args.gen_lr,  epoch, args.epochs, args.min_lr, args.warmup_epochs)
            disc_lr_now = cosine_lr(args.disc_lr, epoch, args.epochs, args.min_lr, args.warmup_epochs)
        else:
            # Linear decay after warmup
            if epoch < args.warmup_epochs:
                gen_lr_now  = args.gen_lr  * ((epoch + 1) / args.warmup_epochs)
                disc_lr_now = args.disc_lr * ((epoch + 1) / args.warmup_epochs)
            else:
                decay_frac  = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
                gen_lr_now  = max(args.gen_lr  * (1.0 - decay_frac), args.min_lr)
                disc_lr_now = max(args.disc_lr * (1.0 - decay_frac), args.min_lr)

        gen_opt.learning_rate.assign(gen_lr_now)
        disc_opt.learning_rate.assign(disc_lr_now)

        # L1 linear decay (100→50 by default)
        l1_frac       = epoch / max(1, args.epochs - 1)
        lambda_l1_now = args.lambda_l1 + (args.lambda_l1_end - args.lambda_l1) * l1_frac

        # Aux-loss warm-up ramp: FM, perceptual, MS-SSIM all start at 0
        # and ramp to full weight over 5 epochs after the LR warmup phase.
        aux_ramp_end = args.warmup_epochs + 5
        if epoch < args.warmup_epochs:
            aux_scale = 0.0
        elif epoch < aux_ramp_end:
            aux_scale = float(epoch - args.warmup_epochs) / 5.0
        else:
            aux_scale = 1.0

        fm_lambda_now      = args.feature_matching_lambda * aux_scale
        perc_lambda_now    = args.perceptual_lambda       * aux_scale
        ms_ssim_lambda_now = args.ms_ssim_lambda          * aux_scale

        # Anneal disc input noise 0.1 → 0 over first 50% of training
        noise_anneal_frac = min(1.0, epoch / max(1, args.epochs * 0.5))
        disc_noise_now    = args.disc_input_noise_std * (1.0 - noise_anneal_frac)

        gen_ls, disc_ls, l1_ls, adv_ls = [], [], [], []
        fm_ls, perc_ls, ms_ssim_ls, r1_ls = [], [], [], []
        demo_done = False
        step = 0

        pbar = tqdm(train_ds, desc=f"Epoch {epoch + 1:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            if step >= steps_per_epoch:
                break

            sat_b, map_b = batch
            update_disc  = ((step % max(1, args.disc_update_interval)) == 0)

            per_replica = strategy.run(
                train_step,
                args=(
                    sat_b, map_b,
                    generator, discriminators,
                    gen_opt, disc_opt,
                    tf.constant(lambda_l1_now,      dtype=tf.float32),
                    args.label_smoothing,
                    args.label_noise,
                    tf.constant(fm_lambda_now,      dtype=tf.float32),
                    perc_model,
                    tf.constant(perc_lambda_now,    dtype=tf.float32),
                    args.gan_mode,
                    update_disc,
                    args.gen_updates,
                    tf.constant(disc_noise_now,     dtype=tf.float32),
                    tf.constant(ms_ssim_lambda_now, dtype=tf.float32),
                    tf.constant(args.r1_gamma,      dtype=tf.float32),
                ),
            )

            _mean = lambda idx: float(strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica[idx], axis=None))

            gt, ga, gl1, dt_, fm_, perc_, ms_, r1_ = [_mean(i) for i in range(8)]

            gen_ls.append(gt);     disc_ls.append(dt_)
            l1_ls.append(gl1);    adv_ls.append(ga)
            fm_ls.append(fm_);    perc_ls.append(perc_)
            ms_ssim_ls.append(ms_); r1_ls.append(r1_)

            pbar.set_postfix({'gen': f'{gt:.4f}', 'disc': f'{dt_:.4f}', 'r1': f'{r1_:.4f}'})

            step += 1
            if args.mode == 'demo' and step >= args.demo_steps:
                demo_done = True
                break

        if demo_done:
            print(f"[DEMO] Stopped after {args.demo_steps} steps.")
            break

        mg  = float(np.mean(gen_ls))
        md  = float(np.mean(disc_ls))
        ml  = float(np.mean(l1_ls))
        ma  = float(np.mean(adv_ls))
        mf  = float(np.mean(fm_ls))
        mp  = float(np.mean(perc_ls))
        mms = float(np.mean(ms_ssim_ls))
        mr1 = float(np.mean(r1_ls))
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"gen={mg:.4f} disc={md:.4f} l1={ml:.4f} ms_ssim={mms:.4f} "
            f"adv={ma:.4f} fm={mf:.4f} perc={mp:.4f} r1={mr1:.4f} | "
            f"gen_lr={gen_lr_now:.2e} disc_lr={disc_lr_now:.2e} | {elapsed:.1f}s"
        )

        if epoch >= 4 and md < 0.05:
            print(f"  [WARN] disc={md:.4f} very low — D nearing collapse; R1 should stabilise it.")
        if epoch >= 4 and md > 0.70:
            print(f"  [WARN] disc={md:.4f} high — G may be failing (check gen_updates or lambda_l1).")

        with sw.as_default():
            tf.summary.scalar('loss/gen',      mg,  step=epoch)
            tf.summary.scalar('loss/disc',     md,  step=epoch)
            tf.summary.scalar('loss/l1',       ml,  step=epoch)
            tf.summary.scalar('loss/adv',      ma,  step=epoch)
            tf.summary.scalar('loss/fm',       mf,  step=epoch)
            tf.summary.scalar('loss/perc',     mp,  step=epoch)
            tf.summary.scalar('loss/ms_ssim',  mms, step=epoch)
            tf.summary.scalar('loss/r1',       mr1, step=epoch)
            tf.summary.scalar('lr/gen',        gen_lr_now,  step=epoch)
            tf.summary.scalar('lr/disc',       disc_lr_now, step=epoch)

        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            save_sample(generator, test_ds, args.results_dir, epoch + 1)

        # Per-epoch CSV row (metrics filled in when eval runs)
        csv_row = {
            'epoch': epoch + 1,
            'g_loss': f'{mg:.6f}', 'd_loss': f'{md:.6f}',
            'l1': f'{ml:.6f}', 'adv': f'{ma:.6f}',
            'fm': f'{mf:.6f}', 'perc': f'{mp:.6f}',
            'ms_ssim_loss': f'{mms:.6f}', 'r1': f'{mr1:.6f}',
            'gen_lr': f'{gen_lr_now:.2e}', 'disc_lr': f'{disc_lr_now:.2e}',
        }

        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate_metrics(generator, test_ds)
            if metrics:
                mae_v, ssim_v, psnr_v = metrics['MAE'], metrics['SSIM'], metrics['PSNR']
                print(
                    f"  [METRICS] MAE={mae_v:.4f} SSIM={ssim_v:.4f} PSNR={psnr_v:.2f}dB"
                )
                with sw.as_default():
                    tf.summary.scalar('metrics/SSIM', ssim_v, step=epoch)
                    tf.summary.scalar('metrics/MAE',  mae_v,  step=epoch)
                    tf.summary.scalar('metrics/PSNR', psnr_v, step=epoch)

                csv_row.update({
                    'mae': f'{mae_v:.6f}',
                    'ssim': f'{ssim_v:.6f}',
                    'psnr': f'{psnr_v:.4f}',
                })

                # Save best model whenever SSIM improves
                if ssim_v > best_ssim:
                    best_ssim = ssim_v
                    bp = os.path.join(args.savedir, 'best_generator.keras')
                    if tf.io.gfile.exists(bp):
                        tf.io.gfile.remove(bp)
                    generator.save(bp)
                    for di, d in enumerate(discriminators):
                        d.save(os.path.join(args.savedir, f'best_discriminator_d{di+1}.keras'))
                    print(f"  [BEST] SSIM={best_ssim:.4f} at epoch {epoch + 1} -> {bp}")

        append_csv_log(csv_path, csv_row)

        # Periodic checkpoint saving (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            periodic_ckpt_path = os.path.join(args.savedir, f'generator_epoch_{epoch + 1:03d}.keras')
            generator.save(periodic_ckpt_path)
            for di, d in enumerate(discriminators):
                d.save(os.path.join(args.savedir, f'discriminator_epoch_{epoch + 1:03d}_d{di+1}.keras'))
            print(f"  [PERIODIC] Saved epoch {epoch + 1:03d} -> {periodic_ckpt_path}")

        if (epoch + 1) % args.save_every == 0:
            epoch_var.assign(epoch + 1)
            print(f"  [CKPT] {mgr.save()}")

    print(f"[DONE] Training completed: Best SSIM={best_ssim:.4f}, epochs={args.epochs}")
    analyze_log(csv_path)

    os.makedirs('outputs', exist_ok=True)
    make_gif(args.results_dir, 'outputs/training_progress.gif')

    # Print resume instructions
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] To continue training from saved checkpoints:")
    hd_flag = ' --use_pix2pixhd' if args.use_pix2pixhd else ''
    print(f"  CUDA_VISIBLE_DEVICES=2 python src/train.py{hd_flag} "
          f"--resume {args.savedir}/best_generator.keras "
          f"--resume_disc {args.savedir}/best_discriminator_d1.keras")
    if args.epochs > 10:
        print(f"  CUDA_VISIBLE_DEVICES=2 python src/train.py{hd_flag} "
              f"--resume {args.savedir}/generator_epoch_050.keras "
              f"--resume_disc {args.savedir}/discriminator_epoch_050_d1.keras")
    print(f"{'=' * 60}\n")

    if args.export:
        ep = os.path.join(args.savedir, 'generator_final.keras')
        if tf.io.gfile.exists(ep):
            tf.io.gfile.remove(ep)
        generator.save(ep)
        print(f"[EXPORT] {ep}")
        print(f"[SUMMARY] To continue from exported model:")
        print(f"  CUDA_VISIBLE_DEVICES=2 python src/train.py --resume {ep}")


if __name__ == '__main__':
    main()
