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
import csv
import argparse
import time
import math
import numpy as np
import tensorflow as tf
import imageio
from pathlib import Path
from tqdm import tqdm

try:
    from tensorflow_addons.layers import SpectralNormalization
    TFA_AVAILABLE = True
except Exception:
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
        if use_cache:
            ds = ds.cache()
        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(
            lambda p: load_test_image(p, split_order),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if use_cache:
            ds = ds.cache()
        ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)

    return ds


# -------------------------------------------------
# MODEL ARCHITECTURE
# -------------------------------------------------

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        c = input_shape[-1]
        self.scale = self.add_weight('scale', (c,), initializer='ones', trainable=True)
        self.offset = self.add_weight('offset', (c,), initializer='zeros', trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset


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
        self._reduced = reduced
        self._channels = channels

    def build(self, input_shape):
        self.gamma = self.add_weight('gamma', shape=(), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        shape = tf.shape(x)
        B, H, W = shape[0], shape[1], shape[2]
        HW = H * W
        reduced = self._reduced
        C = self._channels

        q = tf.reshape(self.q_conv(x), [B, HW, reduced])
        k = tf.reshape(self.k_conv(x), [B, HW, reduced])
        v = tf.reshape(self.v_conv(x), [B, HW, C])

        scale = tf.math.sqrt(tf.cast(reduced, tf.float32))
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale, axis=-1)

        attended = tf.reshape(tf.matmul(attn, v), [B, H, W, C])
        return self.gamma * self.out_conv(attended) + x


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
        block.add(tf.keras.layers.Dropout(0.5))
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


def build_discriminator(use_spectral_norm=False):
    """
    PatchGAN discriminator (70x70 receptive field) with LayerNorm.
    Returns [patch_logits, f1, f2, f3, f4] for feature-matching loss.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
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
    return tf.add_n(losses) / float(len(losses))


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

@tf.function
def train_step(satellite, real_map, generator, discriminator,
               gen_opt, disc_opt, lambda_l1, label_smoothing, label_noise,
               fm_lambda, perc_model, perc_lambda, gan_mode, update_disc,
               gen_updates, disc_input_noise_std, ms_ssim_lambda, r1_gamma):
    """
    Single distributed training step.

    Generator loss  = LSGAN_adv + λ_l1*L1 + λ_ms*MS-SSIM + λ_fm*FM + λ_perc*Perceptual
    Discriminator loss = LSGAN + R1 gradient penalty (prevents collapse)
    """

    def add_noise(x):
        if disc_input_noise_std <= 0:
            return x
        return x + tf.random.normal(tf.shape(x), 0.0, disc_input_noise_std, dtype=x.dtype)

    gen_total = tf.constant(0.0, dtype=tf.float32)
    gen_adv   = tf.constant(0.0, dtype=tf.float32)
    gen_l1    = tf.constant(0.0, dtype=tf.float32)
    fm        = tf.constant(0.0, dtype=tf.float32)
    perc      = tf.constant(0.0, dtype=tf.float32)
    ms_ssim   = tf.constant(0.0, dtype=tf.float32)
    # Initialise gen_map so it is always bound before the discriminator step.
    gen_map   = generator(satellite, training=False)

    # ---- Generator updates ----
    for _ in range(max(1, gen_updates)):
        with tf.GradientTape() as gt:
            gen_map = generator(satellite, training=True)

            disc_real_out = discriminator([satellite, add_noise(real_map)], training=False)
            disc_fake_out = discriminator([satellite, add_noise(gen_map)],  training=False)

            gen_total, gen_adv, gen_l1 = generator_loss(
                disc_fake_out[0], gen_map, real_map, lambda_l1, gan_mode,
            )

            if fm_lambda > 0:
                fm = feature_matching_loss_fn(disc_real_out[1:], disc_fake_out[1:])
                gen_total = gen_total + fm_lambda * fm
            else:
                fm = tf.constant(0.0, dtype=tf.float32)

            perc = perceptual_loss_fn(perc_model, real_map, gen_map)
            gen_total = gen_total + perc_lambda * perc

            if ms_ssim_lambda > 0:
                ms_ssim = ms_ssim_loss_fn(real_map, gen_map)
                gen_total = gen_total + ms_ssim_lambda * ms_ssim
            else:
                ms_ssim = tf.constant(0.0, dtype=tf.float32)

        gen_grads = gt.gradient(gen_total, generator.trainable_variables)
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, 1.0)
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))

    # ---- Discriminator update with R1 gradient penalty ----
    real_map_f32  = tf.cast(real_map, tf.float32)
    gen_map_disc  = tf.stop_gradient(tf.cast(gen_map, tf.float32))

    with tf.GradientTape() as dt:
        # R1 penalty: penalise ||∇_x D(x_real)||² to prevent D from becoming
        # arbitrarily sharp on real inputs, which causes the observed collapse.
        with tf.GradientTape() as r1_tape:
            r1_tape.watch(real_map_f32)
            disc_real_out = discriminator([satellite, add_noise(real_map_f32)], training=True)
            real_patch = tf.cast(disc_real_out[0], tf.float32)
        r1_grads  = r1_tape.gradient(real_patch, real_map_f32)
        r1_penalty = r1_gamma * 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(r1_grads), axis=[1, 2, 3])
        )

        disc_fake_out = discriminator([satellite, add_noise(gen_map_disc)], training=True)
        disc_total = discriminator_loss(
            disc_real_out[0], disc_fake_out[0],
            label_smoothing, label_noise, gan_mode,
        ) + r1_penalty

    disc_grads = dt.gradient(disc_total, discriminator.trainable_variables)
    disc_grads, _ = tf.clip_by_global_norm(disc_grads, 1.0)
    if update_disc:
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

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
    frames = sorted(Path(results_dir).glob("epoch_*.png"))
    if not frames:
        return
    images = [imageio.v2.imread(str(f)) for f in frames]
    imageio.mimsave(output_path, images, fps=4)
    print(f"[GIF] {output_path}")


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
    ap.add_argument('--lr_schedule',    default='cosine', choices=['cosine', 'linear'])

    # Learning rates
    ap.add_argument('--lr',      type=float, default=None,
                    help='Legacy flag: overrides both gen_lr and disc_lr.')
    ap.add_argument('--gen_lr',  type=float, default=2e-4)
    ap.add_argument('--disc_lr', type=float, default=1e-4)
    ap.add_argument('--min_lr',  type=float, default=1e-7,
                    help='Floor LR for cosine annealing.')

    # Loss weights
    ap.add_argument('--lambda_l1',     type=float, default=50.0,
                    help='L1 pixel loss weight at epoch 0.')
    ap.add_argument('--lambda_l1_end', type=float, default=30.0,
                    help='L1 pixel loss weight at final epoch (linear decay).')
    ap.add_argument('--ms_ssim_lambda',             type=float, default=80.0,
                    help='MS-SSIM loss weight (explicit SSIM optimisation).')
    ap.add_argument('--perceptual_lambda', '--lambda_perc',
                    dest='perceptual_lambda',       type=float, default=10.0)
    ap.add_argument('--feature_matching_lambda', '--lambda_fm',
                    dest='feature_matching_lambda', type=float, default=10.0)
    ap.add_argument('--r1_gamma',      type=float, default=10.0,
                    help='R1 gradient-penalty coefficient (0 = disabled).')

    # GAN stability knobs
    ap.add_argument('--gan_mode',           default='lsgan', choices=['lsgan', 'bce'])
    ap.add_argument('--label_smoothing',    type=float, default=0.15)
    ap.add_argument('--label_noise',        type=float, default=0.02)
    ap.add_argument('--disc_update_interval', type=int, default=1)
    ap.add_argument('--gen_updates',        type=int,   default=2,
                    help='Generator updates per discriminator update (TTUR).')
    ap.add_argument('--disc_input_noise_std', type=float, default=0.1,
                    help='Disc input noise std, annealed to 0 over first 50%% of training.')

    # Architecture
    ap.add_argument('--generator_norm', default='instance', choices=['instance', 'batch'])
    ap.add_argument('--spectral_norm',  action='store_true')

    # Dataset
    ap.add_argument('--split_order',   default='map_sat', choices=['map_sat', 'sat_map'])
    ap.add_argument('--cache_dataset', nargs='?', const=True, default=False, type=str2bool)

    # Runtime
    ap.add_argument('--mode',       default='full', choices=['full', 'demo'])
    ap.add_argument('--demo_steps', type=int, default=5)
    ap.add_argument('--restore',    action='store_true', help='Resume from latest checkpoint.')
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
    if not gpus:
        print("[WARN] No GPUs found — training will be slow.")
        if args.require_gpu:
            raise RuntimeError("--require_gpu set but no GPU visible to TensorFlow.")

    if args.multi_gpu and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[GPU] MirroredStrategy over {strategy.num_replicas_in_sync} GPUs")
        effective_batch = args.batch_size * strategy.num_replicas_in_sync
    else:
        if args.multi_gpu and len(gpus) <= 1:
            print("[WARN] --multi_gpu set but ≤1 GPU visible. Using default strategy.")
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
        generator = build_generator(norm_type=args.generator_norm)

        if args.spectral_norm and not TFA_AVAILABLE:
            print("[WARN] --spectral_norm requires tensorflow-addons. Skipping.")
            args.spectral_norm = False
        discriminator = build_discriminator(use_spectral_norm=args.spectral_norm)

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

        gen_opt  = tf.keras.optimizers.Adam(learning_rate=args.gen_lr,  beta_1=0.5)
        disc_opt = tf.keras.optimizers.Adam(learning_rate=args.disc_lr, beta_1=0.5)

        epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(
            generator=generator, discriminator=discriminator,
            gen_opt=gen_opt, disc_opt=disc_opt, epoch=epoch_var,
        )

    mgr = tf.train.CheckpointManager(ckpt, args.savedir, max_to_keep=5)

    start_epoch = 0
    if args.restore and mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint)
        start_epoch = int(epoch_var.numpy())
        print(f"[CKPT] Restored from {mgr.latest_checkpoint} -> resuming from epoch {start_epoch + 1}")

    sw       = tf.summary.create_file_writer(args.logdir)
    csv_path = init_csv_log(args.logdir)
    best_ssim = 0.0

    print(f"\n{'=' * 60}")
    print(f"  Pix2Pix Phase 2 | {args.epochs} epochs | LR={args.lr_schedule}")
    print(f"  L1={args.lambda_l1}→{args.lambda_l1_end} | MS-SSIM={args.ms_ssim_lambda}")
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
                    generator, discriminator,
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
                    bp = os.path.join(args.savedir, 'best_generator')
                    if tf.io.gfile.exists(bp):
                        tf.io.gfile.rmtree(bp)
                    generator.save(bp)
                    print(f"  [BEST] SSIM={best_ssim:.4f} at epoch {epoch + 1} -> {bp}")

        append_csv_log(csv_path, csv_row)

        if (epoch + 1) % args.save_every == 0:
            epoch_var.assign(epoch + 1)
            print(f"  [CKPT] {mgr.save()}")

    print(f"\n[DONE] Best SSIM: {best_ssim:.4f}")
    analyze_log(csv_path)

    os.makedirs('outputs', exist_ok=True)
    make_gif(args.results_dir, 'outputs/training_progress.gif')

    if args.export:
        ep = os.path.join(args.savedir, 'generator_final')
        if tf.io.gfile.exists(ep):
            tf.io.gfile.rmtree(ep)
        generator.save(ep)
        print(f"[EXPORT] {ep}")


if __name__ == '__main__':
    main()
