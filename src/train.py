"""
train.py — Enhanced Pix2Pix Training Script
Fixes applied:
    1. Default paths are repo-relative (no more ../)
    2. Tensor float casting in all logging
    3. Demo early-stop breaks outer loop too
Enhancements:
    - Multi-GPU support via MirroredStrategy (8x A6000 ready)
    - Configurable paired-image split order (map|sat or sat|map)
    - Dataset sanity-check preview image before long training
    - LR decay schedule (paper-faithful: decay from epoch 100)
    - Per-epoch MAE/SSIM/PSNR evaluation
    - TensorBoard logging
    - Best model checkpoint based on SSIM
    - Progress bar via tqdm
    - GIF generation from epoch samples
    - Resumable from any checkpoint
"""

import os
import argparse
import time
import math
import random
import numpy as np
import tensorflow as tf
import imageio
from pathlib import Path
from collections import deque
from tqdm import tqdm
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess_input

# Optional: scikit-image for metrics
try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("[WARN] scikit-image not installed. pip install scikit-image for SSIM/PSNR metrics.")

# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def list_image_files(data_dir):
    """Return sorted list of supported image files in a directory."""
    files = []
    for ext in SUPPORTED_IMAGE_EXTS:
        files.extend(tf.io.gfile.glob(os.path.join(data_dir, ext)))
    return sorted(files)


def map_likeliness_score(img_01):
    """Heuristic map-likeliness score from saturation and colorfulness."""
    hsv = tf.image.rgb_to_hsv(img_01)
    sat_mean = float(tf.reduce_mean(hsv[..., 1]).numpy())

    arr = img_01.numpy()
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + (0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))

    return sat_mean + (0.5 * float(colorfulness))


def suggest_split_order(data_dir, num_samples=50):
    """Suggest map_sat or sat_map by comparing left/right map-likeliness."""
    files = list_image_files(data_dir)[:num_samples]
    if not files:
        return None

    votes_map_sat = 0
    votes_sat_map = 0
    margins = []

    for image_path in files:
        img = tf.io.read_file(image_path)
        # Single decode path for jpg/jpeg/png.
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)

        half_w = tf.shape(img)[1] // 2
        left = img[:, :half_w, :]
        right = img[:, half_w:, :]

        left_score = map_likeliness_score(left)
        right_score = map_likeliness_score(right)
        margins.append(abs(left_score - right_score))

        if left_score >= right_score:
            votes_map_sat += 1
        else:
            votes_sat_map += 1

    suggested = "map_sat" if votes_map_sat >= votes_sat_map else "sat_map"
    confidence = max(votes_map_sat, votes_sat_map) / max(1, len(files))
    mean_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "suggested": suggested,
        "confidence": confidence,
        "votes_map_sat": votes_map_sat,
        "votes_sat_map": votes_sat_map,
        "mean_margin": mean_margin,
        "num_samples": len(files),
    }

def load_image(image_path, split_order="map_sat"):
    """
    Load and split one paired image.

    split_order:
      - map_sat: left=map, right=satellite (common in maps dataset)
      - sat_map: left=satellite, right=map
    """
    img = tf.io.read_file(image_path)
    # Single decode path for jpg/jpeg/png.
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    w = tf.shape(img)[1] // 2

    left = img[:, :w, :]
    right = img[:, w:, :]

    if split_order == "sat_map":
        satellite = left
        real_map = right
    else:
        real_map = left
        satellite = right

    real_map = tf.cast(real_map, tf.float32)
    satellite = tf.cast(satellite, tf.float32)
    return satellite, real_map


def normalize(satellite, real_map):
    """Normalize to [-1, 1]."""
    return (satellite / 127.5) - 1.0, (real_map / 127.5) - 1.0


def random_jitter(satellite, real_map):
    """Augmentation: resize to 286, random crop to 256, random flip."""
    satellite = tf.image.resize(satellite, [286, 286], method='nearest')
    real_map  = tf.image.resize(real_map,  [286, 286], method='nearest')
    # Stack for consistent crop
    stacked = tf.stack([satellite, real_map], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, 256, 256, 3])
    satellite, real_map = cropped[0], cropped[1]
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        satellite = tf.image.flip_left_right(satellite)
        real_map  = tf.image.flip_left_right(real_map)
    return satellite, real_map


def load_train_image(image_path, split_order="map_sat"):
    satellite, real_map = load_image(image_path, split_order=split_order)
    satellite, real_map = random_jitter(satellite, real_map)
    satellite, real_map = normalize(satellite, real_map)
    return satellite, real_map


def load_test_image(image_path, split_order="map_sat"):
    satellite, real_map = load_image(image_path, split_order=split_order)
    satellite = tf.image.resize(satellite, [256, 256])
    real_map  = tf.image.resize(real_map,  [256, 256])
    satellite, real_map = normalize(satellite, real_map)
    return satellite, real_map


def build_dataset(data_dir, batch_size, is_train=True, buffer_size=400, split_order="map_sat", use_cache=True):
    image_files = list_image_files(data_dir)
    if not image_files:
        raise RuntimeError(
            f"No images found in {data_dir}. Supported extensions: {', '.join(SUPPORTED_IMAGE_EXTS)}"
        )

    files = tf.data.Dataset.from_tensor_slices(image_files)
    if is_train:
        files = files.shuffle(len(image_files), reshuffle_each_iteration=True)

    if is_train:
        ds = files.map(
            lambda p: load_train_image(p, split_order=split_order),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if use_cache:
            ds = ds.cache()
        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = files.map(
            lambda p: load_test_image(p, split_order=split_order),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if use_cache:
            ds = ds.cache()
        ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds


def save_data_check_preview(data_dir, split_order, output_path):
    """Save one preview panel showing the current split interpretation."""
    files = list_image_files(data_dir)
    if not files:
        print(f"[WARN] Data check skipped: no files in {data_dir}")
        return

    sat, real_map = load_test_image(files[0], split_order=split_order)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow((sat.numpy() * 0.5 + 0.5).clip(0, 1))
    axes[0].set_title('Input (Satellite)')
    axes[0].axis('off')
    axes[1].imshow((real_map.numpy() * 0.5 + 0.5).clip(0, 1))
    axes[1].set_title('Target (Map)')
    axes[1].axis('off')
    plt.tight_layout()

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[DATA] Saved split-order preview to {output_path} (split_order={split_order})")


def save_random_split_samples(data_dir, split_order, output_path, num_samples=5):
    """Save N random [input,target] sample pairs for manual alignment checks."""
    files = list_image_files(data_dir)
    if not files:
        print(f"[WARN] Random sample preview skipped: no files in {data_dir}")
        return

    chosen = random.sample(files, k=min(num_samples, len(files)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for i, img_path in enumerate(chosen):
        sat, real_map = load_test_image(img_path, split_order=split_order)
        axes[i, 0].imshow((sat.numpy() * 0.5 + 0.5).clip(0, 1))
        axes[i, 0].set_title(f'Input (Satellite) #{i+1}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow((real_map.numpy() * 0.5 + 0.5).clip(0, 1))
        axes[i, 1].set_title(f'Target (Map) #{i+1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[DATA] Saved random pair preview to {output_path} ({len(chosen)} samples)")


class InstanceNormalization(tf.keras.layers.Layer):
    """Simple Instance Normalization for generator stability."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.scale = self.add_weight(
            name='scale', shape=(channels,), initializer='ones', trainable=True
        )
        self.offset = self.add_weight(
            name='offset', shape=(channels,), initializer='zeros', trainable=True
        )

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(var + self.epsilon)
        return self.scale * normalized + self.offset


def build_perceptual_model(layer_names=("block3_conv3", "block4_conv3")):
    """Create frozen VGG19 feature extractor for perceptual loss."""
    base = VGG19(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    outputs = [base.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=base.input, outputs=outputs, name="vgg19_perceptual")
    model.trainable = False
    return model


def preprocess_for_vgg(x):
    """Convert normalized [-1,1] tensor to VGG19 expected input format."""
    x = tf.cast(x, tf.float32)
    x = (x + 1.0) * 127.5
    x = tf.image.resize(x, [224, 224])
    return vgg_preprocess_input(x)


def perceptual_loss(perceptual_model, real_img, fake_img):
    """L1 distance in VGG feature space."""
    if perceptual_model is None:
        return tf.constant(0.0, dtype=tf.float32)

    real_feats = perceptual_model(preprocess_for_vgg(real_img), training=False)
    fake_feats = perceptual_model(preprocess_for_vgg(fake_img), training=False)

    if not isinstance(real_feats, (list, tuple)):
        real_feats = [real_feats]
        fake_feats = [fake_feats]

    losses = []
    for real_f, fake_f in zip(real_feats, fake_feats):
        losses.append(tf.reduce_mean(tf.abs(tf.stop_gradient(real_f) - fake_f)))
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def _norm_layer(norm_type='batch'):
    if norm_type == 'instance':
        return InstanceNormalization()
    return tf.keras.layers.BatchNormalization()


def residual_block(x, filters, norm_type='instance'):
    """Simple residual bottleneck block."""
    init = tf.random_normal_initializer(0., 0.02)
    skip = x
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer=init, use_bias=False)(x)
    y = _norm_layer(norm_type)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer=init, use_bias=False)(y)
    y = _norm_layer(norm_type)(y)
    y = tf.keras.layers.Add()([skip, y])
    return tf.keras.layers.ReLU()(y)


def downsample(filters, size, apply_norm=True, norm_type='batch'):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=init, use_bias=False))
    if apply_norm:
        block.add(_norm_layer(norm_type))
    block.add(tf.keras.layers.LeakyReLU())
    return block


def upsample(filters, size, apply_dropout=False, norm_type='batch'):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.UpSampling2D(size=2, interpolation='nearest'))
    block.add(tf.keras.layers.Conv2D(
        filters, size, strides=1, padding='same',
        kernel_initializer=init, use_bias=False))
    block.add(_norm_layer(norm_type))
    if apply_dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block


def build_generator(norm_type='instance', num_res_blocks=2):
    """U-Net generator with skip connections."""
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    init = tf.random_normal_initializer(0., 0.02)

    # Encoder (downsampling)
    down_stack = [
        downsample(64,  4, apply_norm=False, norm_type=norm_type),  # (128, 128, 64)
        downsample(128, 4, norm_type=norm_type),                    # (64, 64, 128)
        downsample(256, 4, norm_type=norm_type),                    # (32, 32, 256)
        downsample(512, 4, norm_type=norm_type),                    # (16, 16, 512)
        downsample(512, 4, norm_type=norm_type),                    # (8, 8, 512)
        downsample(512, 4, norm_type=norm_type),                    # (4, 4, 512)
        downsample(512, 4, norm_type=norm_type),                    # (2, 2, 512)
        downsample(512, 4, norm_type=norm_type),                    # (1, 1, 512)
    ]
    # Decoder (upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),   # (2, 2, 1024)
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),   # (4, 4, 1024)
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),   # (8, 8, 1024)
        upsample(512, 4, norm_type=norm_type),                       # (16, 16, 1024)
        upsample(256, 4, norm_type=norm_type),                       # (32, 32, 512)
        upsample(128, 4, norm_type=norm_type),                       # (64, 64, 256)
        upsample(64,  4, norm_type=norm_type),                       # (128, 128, 128)
    ]
    last = tf.keras.Sequential([
        tf.keras.layers.UpSampling2D(size=2, interpolation='nearest'),
        tf.keras.layers.Conv2D(3, 4, strides=1, padding='same',
                               kernel_initializer=init, activation='tanh', dtype='float32')
    ])

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for _ in range(max(0, num_res_blocks)):
        x = residual_block(x, filters=512, norm_type=norm_type)

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def build_discriminator():
    """PatchGAN discriminator (70x70 patches)."""
    init = tf.random_normal_initializer(0., 0.02)
    inp    = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    target = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.Concatenate()([inp, target])   # (256, 256, 6)
    f1 = downsample(64,  4, apply_norm=False, norm_type='batch')(x)  # (128, 128, 64)
    f2 = downsample(128, 4, norm_type='batch')(f1)                    # (64, 64, 128)
    f3 = downsample(256, 4, norm_type='batch')(f2)                    # (32, 32, 256)
    x = tf.keras.layers.ZeroPadding2D()(f3)                           # (34, 34, 256)
    x = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=init, use_bias=False)(x)  # (31, 31, 512)
    x = tf.keras.layers.BatchNormalization()(x)
    f4 = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(f4)
    patch = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=init)(x)  # (30, 30, 1)
    return tf.keras.Model(inputs=[inp, target], outputs=[patch, f1, f2, f3, f4])


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100, gan_mode='lsgan'):
    """Adversarial + L1 loss."""
    if gan_mode == 'lsgan':
        adv_loss = tf.reduce_mean(tf.square(disc_generated_output - 1.0))
    else:
        adv_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss  = tf.reduce_mean(tf.abs(target - gen_output))
    total    = adv_loss + (lambda_l1 * l1_loss)
    return total, adv_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output, label_smoothing=0.0, label_noise=0.0, gan_mode='lsgan'):
    """BCE on real + fake with optional one-sided label smoothing."""
    real_target = tf.ones_like(disc_real_output) * (1.0 - label_smoothing)
    fake_target = tf.zeros_like(disc_generated_output)

    if label_noise > 0.0:
        real_noise = tf.random.uniform(tf.shape(real_target), minval=-label_noise, maxval=label_noise)
        fake_noise = tf.random.uniform(tf.shape(fake_target), minval=-label_noise, maxval=label_noise)
        real_target = tf.clip_by_value(real_target + real_noise, 0.0, 1.0)
        fake_target = tf.clip_by_value(fake_target + fake_noise, 0.0, 1.0)

    if gan_mode == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(disc_real_output - real_target))
        fake_loss = tf.reduce_mean(tf.square(disc_generated_output - fake_target))
        return 0.5 * (real_loss + fake_loss)

    real_loss = loss_object(real_target, disc_real_output)
    fake_loss = loss_object(fake_target, disc_generated_output)
    return real_loss + fake_loss


def feature_matching_loss(real_features, fake_features):
    """L1 feature matching across discriminator intermediate activations."""
    losses = []
    for real_f, fake_f in zip(real_features, fake_features):
        losses.append(tf.reduce_mean(tf.abs(tf.stop_gradient(real_f) - fake_f)))
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)


# ─────────────────────────────────────────────
# TRAIN STEP
# ─────────────────────────────────────────────

def train_step(satellite, real_map, generator, discriminator,
               gen_optimizer, disc_optimizer, lambda_l1, label_smoothing=0.0,
               label_noise=0.0, fm_lambda=10.0, perceptual_model=None,
               perceptual_lambda=10.0, gan_mode='lsgan', update_discriminator=True):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_map = generator(satellite, training=True)

        disc_real_outputs = discriminator([satellite, real_map], training=True)
        disc_fake_outputs = discriminator([satellite, gen_map],  training=True)

        disc_real = disc_real_outputs[0]
        disc_fake = disc_fake_outputs[0]

        fm_loss = feature_matching_loss(disc_real_outputs[1:], disc_fake_outputs[1:])

        gen_total, gen_adv, gen_l1 = generator_loss(
            disc_fake,
            gen_map,
            real_map,
            lambda_l1,
            gan_mode=gan_mode,
        )
        perc_loss = perceptual_loss(perceptual_model, real_map, gen_map)
        gen_total = gen_total + (fm_lambda * fm_loss) + (perceptual_lambda * perc_loss)
        disc_total = discriminator_loss(
            disc_real,
            disc_fake,
            label_smoothing=label_smoothing,
            label_noise=label_noise,
            gan_mode=gan_mode,
        )

    gen_grads  = gen_tape.gradient(gen_total,  generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_total, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads,  generator.trainable_variables))
    if update_discriminator:
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_total, gen_adv, gen_l1, disc_total, fm_loss, perc_loss


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_metrics(generator, test_ds, num_samples=50):
    """Compute MAE, SSIM, PSNR on test set."""
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


def evaluate(generator, test_ds, num_samples=50):
    """Compatibility wrapper: compute and print MAE/SSIM/PSNR."""
    metrics = evaluate_metrics(generator, test_ds, num_samples=num_samples)
    if not metrics:
        print("[WARN] Metrics unavailable. Install scikit-image for SSIM/PSNR.")
        return {}
    print(
        f"MAE: {metrics['MAE']:.4f} | "
        f"SSIM: {metrics['SSIM']:.4f} | "
        f"PSNR: {metrics['PSNR']:.2f}dB"
    )
    return metrics


# ─────────────────────────────────────────────
# SAMPLE IMAGE SAVING
# ─────────────────────────────────────────────

def save_sample(generator, test_ds, results_dir, epoch):
    """Save a comparison grid: satellite | generated map | ground truth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for sat, real_map in test_ds.take(1):
        gen_map = generator(sat, training=False)
        gen_std = float(tf.math.reduce_std(gen_map[0]).numpy())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Satellite Input', 'Generated Map', 'Ground Truth']
        imgs   = [sat[0], gen_map[0], real_map[0]]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow((img.numpy() * 0.5 + 0.5).clip(0, 1))
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        plt.suptitle(f'Epoch {epoch:03d}', fontsize=14)
        plt.tight_layout()
        path = os.path.join(results_dir, f'epoch_{epoch:03d}.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        return path, gen_std


def make_gif(results_dir, output_path):
    """Compile all epoch PNGs into a training progress GIF."""
    frames = sorted(Path(results_dir).glob("epoch_*.png"))
    if not frames:
        return
    images = [imageio.imread(str(f)) for f in frames]
    imageio.mimsave(output_path, images, fps=4)
    print(f"[GIF] Saved training progress GIF to {output_path}")


def save_saved_model(model, path):
    """Save TensorFlow SavedModel, replacing existing directory if needed."""
    if tf.io.gfile.exists(path):
        tf.io.gfile.rmtree(path)
    tf.saved_model.save(model, path)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in {"true", "1", "yes", "y", "on"}:
            return True
        if value in {"false", "0", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError("Expected true/false for boolean option")

    parser = argparse.ArgumentParser(description="Pix2Pix Satellite→Map Training")
    # Paths — FIX 1: repo-relative defaults (no more ../)
    parser.add_argument('--data_dir',    default='data/train',           help='Training images dir')
    parser.add_argument('--test_dir',    default='data/test',            help='Test images dir')
    parser.add_argument('--results_dir', default='outputs/test_results', help='Sample output dir')
    parser.add_argument('--savedir',     default='saved_models',         help='Checkpoint dir')
    parser.add_argument('--logdir',      default='logs',                 help='TensorBoard log dir')
    # Training hyperparams
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--lambda_l1',   type=float, default=100.0,
                        help='L1 loss weight. Start at 100; try 50 if blurry or 150-200 if noisy.')
    parser.add_argument('--lambda_l1_end', type=float, default=50.0,
                        help='Final L1 weight for linear decay from lambda_l1 to lambda_l1_end')
    parser.add_argument('--decay_epoch', type=int,   default=100,
                        help='Epoch to start LR linear decay (paper default: 100)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Linear LR warmup epochs at start')
    parser.add_argument('--save_every',  type=int,   default=10)
    parser.add_argument('--eval_every',  type=int,   default=10)
    parser.add_argument('--sample_every', type=int, default=1,
                        help='Save sample visualization every N epochs')
    parser.add_argument('--split_order', type=str, default='map_sat', choices=['map_sat', 'sat_map'],
                        help='How paired image is arranged: map_sat means left=map,right=satellite; sat_map means left=satellite,right=map')
    parser.add_argument('--auto_split_detect', action='store_true',
                        help='Analyze dataset halves and suggest split order before training')
    parser.add_argument('--split_confidence_threshold', type=float, default=0.75,
                        help='Confidence threshold for split-order mismatch abort')
    parser.add_argument('--abort_on_split_mismatch', nargs='?', const=True, default=True, type=str2bool,
                        help='Abort if auto-detected split disagrees with selected split_order')
    parser.add_argument('--require_gpu', action='store_true',
                        help='Fail fast if no GPU is visible')
    parser.add_argument('--mixed_precision', nargs='?', const=True, default=True, type=str2bool,
                        help='Enable mixed precision on GPU (true/false)')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='One-sided real-label smoothing for discriminator (0.0-0.2)')
    parser.add_argument('--label_noise', type=float, default=0.02,
                        help='Small random label noise for discriminator targets (0.0-0.1)')
    parser.add_argument('--feature_matching_lambda', type=float, default=10.0,
                        help='Feature matching loss weight for generator')
    parser.add_argument('--perceptual_lambda', type=float, default=10.0,
                        help='Perceptual loss weight (VGG19 block3_conv3 + block4_conv3)')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['lsgan', 'bce'],
                        help='Adversarial objective: lsgan (recommended) or bce')
    parser.add_argument('--disc_update_interval', type=int, default=2,
                        help='Update discriminator every N steps (2 means less frequent D updates)')
    parser.add_argument('--generator_norm', type=str, default='instance', choices=['instance', 'batch'],
                        help='Normalization type in generator blocks')
    parser.add_argument('--res_blocks', type=int, default=2,
                        help='Residual blocks in generator bottleneck')
    parser.add_argument('--cache_dataset', nargs='?', const=True, default=True, type=str2bool,
                        help='Cache mapped dataset in memory (true/false)')
    parser.add_argument('--skip_data_check', nargs='?', const=True, default=False, type=str2bool,
                        help='Skip writing data split sanity preview image (true/false)')
    # Mode
    parser.add_argument('--mode',        default='full', choices=['full', 'demo'])
    parser.add_argument('--demo_steps',  type=int,   default=5)
    parser.add_argument('--restore',     action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--export',      action='store_true', help='Export SavedModel after training')
    # Multi-GPU
    parser.add_argument('--multi_gpu',   action='store_true',
                        help='Enable MirroredStrategy for multi-GPU training')
    args = parser.parse_args()

    if args.batch_size < 4:
        print(f"[WARN] batch_size={args.batch_size} is too low for stable Pix2Pix training; using 4")
        args.batch_size = 4
    elif args.batch_size < 8:
        print(f"[WARN] batch_size={args.batch_size} works, but 8 is usually better if GPU memory allows")

    args.label_smoothing = max(0.0, min(float(args.label_smoothing), 0.2))
    args.label_noise = max(0.0, min(float(args.label_noise), 0.1))
    args.disc_update_interval = max(1, int(args.disc_update_interval))
    args.warmup_epochs = max(0, int(args.warmup_epochs))
    args.sample_every = max(1, int(args.sample_every))
    args.lambda_l1_end = float(args.lambda_l1_end)
    args.perceptual_lambda = max(0.0, float(args.perceptual_lambda))
    args.res_blocks = max(0, min(int(args.res_blocks), 6))

    if args.lambda_l1_end > args.lambda_l1:
        print(
            f"[WARN] lambda_l1_end ({args.lambda_l1_end}) is greater than lambda_l1 ({args.lambda_l1}); "
            "using constant lambda_l1"
        )
        args.lambda_l1_end = args.lambda_l1

    if args.auto_split_detect:
        suggestion = suggest_split_order(args.data_dir, num_samples=50)
        if suggestion is not None:
            print(
                f"[SPLIT] Suggested={suggestion['suggested']} "
                f"confidence={suggestion['confidence']:.2f} "
                f"votes(map_sat/sat_map)={suggestion['votes_map_sat']}/{suggestion['votes_sat_map']} "
                f"mean_margin={suggestion['mean_margin']:.4f}"
            )
            if suggestion['suggested'] != args.split_order:
                print(
                    f"[WARN] Selected split_order={args.split_order} differs from suggested={suggestion['suggested']}"
                )
                if args.abort_on_split_mismatch and suggestion['confidence'] >= args.split_confidence_threshold:
                    raise RuntimeError(
                        "Split-order mismatch detected with high confidence. "
                        f"Suggested={suggestion['suggested']} selected={args.split_order}. "
                        "Fix --split_order and rerun."
                    )

    # ── Create output directories ──────────────────────────────────────
    for d in [args.results_dir, args.savedir, args.logdir]:
        os.makedirs(d, exist_ok=True)

    # ── Multi-GPU Strategy ─────────────────────────────────────────────
    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        num_gpus = strategy.num_replicas_in_sync
        print(f"[GPU] MirroredStrategy: {num_gpus} GPU(s) detected")
        # Scale batch size linearly with GPU count
        effective_batch = args.batch_size * num_gpus
        print(f"[GPU] Effective batch size: {effective_batch} (scaled from {args.batch_size})")
    else:
        strategy = tf.distribute.get_strategy()
        effective_batch = args.batch_size
        num_gpus = 1

    visible_gpus = tf.config.list_physical_devices('GPU')
    if visible_gpus:
        print(f"[GPU] Visible GPUs: {len(visible_gpus)}")
    else:
        print("[WARN] No GPU visible. CPU training will converge slower and often underperform.")
        if args.require_gpu:
            raise RuntimeError("--require_gpu was set, but no GPU is visible.")

    if args.mixed_precision and visible_gpus:
        mixed_precision.set_global_policy('mixed_float16')
        print("[AMP] Mixed precision enabled (mixed_float16)")
    else:
        mixed_precision.set_global_policy('float32')

    if not args.skip_data_check:
        save_data_check_preview(
            args.data_dir,
            args.split_order,
            os.path.join('outputs', 'data_split_preview.png'),
        )
        save_random_split_samples(
            args.data_dir,
            args.split_order,
            os.path.join('outputs', 'data_split_samples.png'),
            num_samples=5,
        )

    # ── Build datasets ─────────────────────────────────────────────────
    train_ds = build_dataset(
        args.data_dir,
        effective_batch,
        is_train=True,
        split_order=args.split_order,
        use_cache=args.cache_dataset,
    )
    test_ds = build_dataset(
        args.test_dir,
        1,
        is_train=False,
        split_order=args.split_order,
        use_cache=args.cache_dataset,
    )

    # Count training samples
    n_train = len(list_image_files(args.data_dir))
    if n_train == 0:
        raise RuntimeError(
            f"No training images found in {args.data_dir}. Supported extensions: {', '.join(SUPPORTED_IMAGE_EXTS)}"
        )
    steps_per_epoch = max(1, int(math.ceil(n_train / effective_batch)))
    print(f"[DATA] {n_train} training images | {steps_per_epoch} steps/epoch")

    # ── Build models inside strategy scope ────────────────────────────
    with strategy.scope():
        generator     = build_generator(norm_type=args.generator_norm, num_res_blocks=args.res_blocks)
        discriminator = build_discriminator()
        perceptual_model = None
        if args.perceptual_lambda > 0.0:
            try:
                perceptual_model = build_perceptual_model()
                print("[LOSS] VGG19 perceptual loss enabled")
            except Exception as ex:
                print(f"[WARN] Could not initialize VGG19 perceptual model: {ex}")
                print("[WARN] Continuing with perceptual_lambda=0.0")
                args.perceptual_lambda = 0.0

        # Optimizers with scalar float LR; decay updates assign into optimizer lr.
        gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=float(args.lr), beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.lr), beta_1=0.5)

    @tf.function
    def single_train_step(sat_batch, map_batch, lambda_l1_value, update_disc):
        return train_step(
            sat_batch,
            map_batch,
            generator,
            discriminator,
            gen_optimizer,
            disc_optimizer,
            lambda_l1_value,
            args.label_smoothing,
            args.label_noise,
            args.feature_matching_lambda,
            perceptual_model,
            args.perceptual_lambda,
            args.gan_mode,
            update_disc,
        )

    if args.multi_gpu:
        distributed_train_ds = strategy.experimental_distribute_dataset(train_ds)

        @tf.function
        def distributed_train_step(dist_batch, lambda_l1_value, update_disc):
            def replica_step(sat_batch, map_batch):
                return train_step(
                    sat_batch,
                    map_batch,
                    generator,
                    discriminator,
                    gen_optimizer,
                    disc_optimizer,
                    lambda_l1_value,
                    args.label_smoothing,
                    args.label_noise,
                    args.feature_matching_lambda,
                    perceptual_model,
                    args.perceptual_lambda,
                    args.gan_mode,
                    update_disc,
                )

            per_replica = strategy.run(replica_step, args=dist_batch)
            gen_total = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None)
            gen_adv = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None)
            gen_l1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None)
            disc_total = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None)
            fm_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[4], axis=None)
            perc_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[5], axis=None)
            return gen_total, gen_adv, gen_l1, disc_total, fm_loss, perc_loss
    else:
        distributed_train_ds = train_ds

    # ── Checkpointing ──────────────────────────────────────────────────
    checkpoint = tf.train.Checkpoint(
        generator=generator, discriminator=discriminator,
        gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, args.savedir, max_to_keep=5)

    start_epoch = 0
    if args.restore and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        # Infer start epoch from checkpoint name
        ckpt_name = os.path.basename(manager.latest_checkpoint)
        try:
            start_epoch = int(ckpt_name.split('-')[-1]) * args.save_every
        except Exception:
            start_epoch = 0
        print(f"[CKPT] Restored from {manager.latest_checkpoint} (epoch ~{start_epoch})")

    # ── TensorBoard summary writer ─────────────────────────────────────
    summary_writer = tf.summary.create_file_writer(args.logdir)

    # ── Training loop ──────────────────────────────────────────────────
    best_ssim = 0.0
    gif_frames = []
    collapse_hits = 0
    gen_loss_window = deque(maxlen=6)

    print(f"\n{'='*60}")
    print(f"  Pix2Pix Training — {args.epochs} epochs")
    print(f"  L1 weight: {args.lambda_l1} -> {args.lambda_l1_end} | LR: {args.lr}")
    print(f"  LR decay starts at epoch: {args.decay_epoch}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # LR warmup + linear decay
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            warmup_lr = args.lr * ((epoch + 1) / args.warmup_epochs)
            gen_optimizer.learning_rate.assign(warmup_lr)
            disc_optimizer.learning_rate.assign(warmup_lr)
        elif epoch >= args.decay_epoch:
            decay_steps = args.epochs - args.decay_epoch
            new_lr = args.lr * (1.0 - (epoch - args.decay_epoch) / max(1, decay_steps))
            new_lr = max(new_lr, 1e-7)
            gen_optimizer.learning_rate.assign(new_lr)
            disc_optimizer.learning_rate.assign(new_lr)

        current_lr = float(gen_optimizer.learning_rate.numpy())
        l1_progress = epoch / max(1, args.epochs - 1)
        lambda_l1_current = args.lambda_l1 + (args.lambda_l1_end - args.lambda_l1) * l1_progress
        lambda_l1_current = float(max(args.lambda_l1_end, lambda_l1_current))

        # ── Batch loop ────────────────────────────────────────────────
        gen_losses, disc_losses = [], []
        gen_l1_losses = []
        gen_adv_losses = []
        fm_losses = []
        perc_losses = []
        demo_done = False  # FIX 3: flag to break outer loop in demo mode

        pbar = tqdm(enumerate(distributed_train_ds.take(steps_per_epoch)), total=steps_per_epoch,
                    desc=f"Epoch {epoch+1:03d}/{args.epochs}", leave=False)

        for step, batch in pbar:
            update_disc = ((step + 1) % args.disc_update_interval == 0)
            if args.multi_gpu:
                gen_total, gen_adv, gen_l1, disc_total, fm_loss, perc_loss = distributed_train_step(
                    batch, tf.constant(lambda_l1_current, dtype=tf.float32), update_disc
                )
            else:
                sat_batch, map_batch = batch
                gen_total, gen_adv, gen_l1, disc_total, fm_loss, perc_loss = single_train_step(
                    sat_batch,
                    map_batch,
                    tf.constant(lambda_l1_current, dtype=tf.float32),
                    update_disc,
                )
            # FIX 2: cast tensors to float before string formatting
            gen_losses.append(float(gen_total))
            disc_losses.append(float(disc_total))
            gen_l1_losses.append(float(gen_l1))
            gen_adv_losses.append(float(gen_adv))
            fm_losses.append(float(fm_loss))
            perc_losses.append(float(perc_loss))

            pbar.set_postfix({
                'gen': f'{float(gen_total):.4f}',
                'disc': f'{float(disc_total):.4f}',
                'l1': f'{float(gen_l1):.4f}',
            })

            # Demo mode early stop
            if args.mode == 'demo' and step + 1 >= args.demo_steps:
                demo_done = True
                break

        if demo_done:
            print(f"[DEMO] Stopping after {args.demo_steps} steps.")
            break  # FIX 3: breaks outer epoch loop too

        # Epoch-level stats
        mean_gen  = float(np.mean(gen_losses))
        mean_disc = float(np.mean(disc_losses))
        mean_l1   = float(np.mean(gen_l1_losses))
        mean_adv  = float(np.mean(gen_adv_losses))
        mean_fm   = float(np.mean(fm_losses))
        mean_perc = float(np.mean(perc_losses))
        gan_l1_ratio = mean_adv / max(1e-8, lambda_l1_current * mean_l1)
        elapsed   = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"gen={mean_gen:.4f} disc={mean_disc:.4f} l1={mean_l1:.4f} "
              f"adv={mean_adv:.4f} fm={mean_fm:.4f} perc={mean_perc:.4f} ratio={gan_l1_ratio:.4f} | "
              f"lr={current_lr:.2e} lambda={lambda_l1_current:.1f} | {elapsed:.1f}s")

        if (epoch + 1) >= 5 and mean_disc < 0.10:
            print(f"[WARN] Discriminator loss is very low ({mean_disc:.4f}); D may be overpowering G.")

        # ── TensorBoard ───────────────────────────────────────────────
        with summary_writer.as_default():
            tf.summary.scalar('loss/generator',     mean_gen,  step=epoch)
            tf.summary.scalar('loss/discriminator', mean_disc, step=epoch)
            tf.summary.scalar('loss/l1',            mean_l1,   step=epoch)
            tf.summary.scalar('loss/adv',           mean_adv,  step=epoch)
            tf.summary.scalar('loss/feature_matching', mean_fm, step=epoch)
            tf.summary.scalar('loss/perceptual',    mean_perc, step=epoch)
            tf.summary.scalar('loss/gan_l1_ratio',  gan_l1_ratio, step=epoch)
            tf.summary.scalar('lr/generator',       current_lr, step=epoch)
            tf.summary.scalar('lambda/l1',          lambda_l1_current, step=epoch)

        # ── Save sample image ─────────────────────────────────────────
        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            result = save_sample(generator, test_ds, args.results_dir, epoch + 1)
            if result:
                frame_path, gen_std = result
                gif_frames.append(frame_path)
                if gen_std < 0.015:
                    collapse_hits += 1
                    print(f"[WARN] Low output variance detected (std={gen_std:.5f})")
                    if collapse_hits >= 3 and (epoch + 1) >= 10:
                        print("[ALERT] Model collapse suspected: generated outputs look nearly constant.")

        # ── Evaluate metrics ──────────────────────────────────────────
        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate_metrics(generator, test_ds, num_samples=50)
            if metrics:
                print(f"  [METRICS] MAE={metrics['MAE']:.4f} | "
                      f"SSIM={metrics['SSIM']:.4f} | PSNR={metrics['PSNR']:.2f}dB")
                with summary_writer.as_default():
                    tf.summary.scalar('metrics/MAE',  metrics['MAE'],  step=epoch)
                    tf.summary.scalar('metrics/SSIM', metrics['SSIM'], step=epoch)
                    tf.summary.scalar('metrics/PSNR', metrics['PSNR'], step=epoch)

                # Save best model based on SSIM
                if metrics['SSIM'] > best_ssim:
                    best_ssim = metrics['SSIM']
                    best_path = os.path.join(args.savedir, 'best_generator')
                    save_saved_model(generator, best_path)
                    print(f"  [BEST] New best SSIM={best_ssim:.4f} → saved to {best_path}")

                if (epoch + 1) >= 10 and metrics['SSIM'] < 0.10:
                    print(
                        f"[ALERT] SSIM is still low ({metrics['SSIM']:.4f}) at epoch {epoch+1}. "
                        "Likely split/data mismatch or collapsed training."
                    )

        gen_loss_window.append(mean_gen)
        if len(gen_loss_window) == gen_loss_window.maxlen:
            if (max(gen_loss_window) - min(gen_loss_window)) < 0.02 and (epoch + 1) >= 15:
                new_lr = max(current_lr * 0.5, 1e-7)
                if new_lr < current_lr:
                    gen_optimizer.learning_rate.assign(new_lr)
                    disc_optimizer.learning_rate.assign(new_lr)
                    print(f"[ADAPT] Generator loss stagnation detected. Lowering LR to {new_lr:.2e}")

        # ── Checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = manager.save()
            print(f"  [CKPT] Saved: {ckpt_path}")

    # ── Post-training ──────────────────────────────────────────────────
    print(f"\n[DONE] Training complete. Best SSIM: {best_ssim:.4f}")

    # Make GIF
    gif_path = os.path.join('outputs', 'training_progress.gif')
    os.makedirs('outputs', exist_ok=True)
    make_gif(args.results_dir, gif_path)

    # Export final SavedModel
    if args.export:
        export_path = os.path.join(args.savedir, 'generator_final')
        save_saved_model(generator, export_path)
        print(f"[EXPORT] Generator exported to {export_path}")
        print(f"[EXPORT] Run inference with: python src/inference.py --model_dir {export_path}")


if __name__ == '__main__':
    main()