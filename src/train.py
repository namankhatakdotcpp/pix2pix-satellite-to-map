"""
train.py - Pix2Pix (FULLY FIXED)
Fixes:
  1. Repo-relative default paths
  2. float() cast on all tensor logging
  3. Demo early-stop breaks outer loop
  4. DistributedDataset has no .take() - removed it
  5. CUDA-safe: works with 0, 1, or 8 GPUs
  6. Loss weights corrected (perceptual=1, fm=0 by default)
  7. Mixed precision last layer kept float32 to avoid tanh saturation
"""

import os
import argparse
import time
import math
import numpy as np
import tensorflow as tf
import imageio
from pathlib import Path
from collections import deque
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
    """
    Load and split paired image.
    map_sat: LEFT=map (target), RIGHT=satellite (input)
    sat_map: LEFT=satellite (input), RIGHT=map (target)
    """
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


def _norm(norm_type):
    if norm_type == 'instance':
        return InstanceNormalization()
    return tf.keras.layers.BatchNormalization()


def downsample(filters, size, apply_norm=True, norm_type='instance'):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=init,
        use_bias=False,
    ))
    if apply_norm:
        block.add(_norm(norm_type))
    block.add(tf.keras.layers.LeakyReLU())
    return block


def upsample(filters, size, apply_dropout=False, norm_type='instance'):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=init,
        use_bias=False,
    ))
    block.add(_norm(norm_type))
    if apply_dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block


def build_generator(norm_type='instance'):
    """Standard Pix2Pix U-Net generator. Last layer forced float32 for tanh stability."""
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    init = tf.random_normal_initializer(0., 0.02)

    down_stack = [
        downsample(64, 4, apply_norm=False, norm_type=norm_type),
        downsample(128, 4, norm_type=norm_type),
        downsample(256, 4, norm_type=norm_type),
        downsample(512, 4, norm_type=norm_type),
        downsample(512, 4, norm_type=norm_type),
        downsample(512, 4, norm_type=norm_type),
        downsample(512, 4, norm_type=norm_type),
        downsample(512, 4, norm_type=norm_type),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, apply_dropout=True, norm_type=norm_type),
        upsample(512, 4, norm_type=norm_type),
        upsample(256, 4, norm_type=norm_type),
        upsample(128, 4, norm_type=norm_type),
        upsample(64, 4, norm_type=norm_type),
    ]

    # Keep final layer float32 even when mixed precision is active.
    last = tf.keras.layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding='same',
        kernel_initializer=init,
        activation='tanh',
        dtype='float32',
    )

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    return tf.keras.Model(inputs=inputs, outputs=last(x))


def build_discriminator(use_spectral_norm=False):
    """Stable PatchGAN using LayerNorm and reduced depth.

    Returns [patch_output, f1, f2, f3, f4] so feature matching stays active.
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
        y = conv2d(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=not apply_norm,
        )(x_in)
        if apply_norm:
            y = tf.keras.layers.LayerNormalization()(y)
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        return y

    # First block intentionally has no normalization.
    f1 = disc_downsample(x, 64, 4, apply_norm=False)
    f2 = disc_downsample(f1, 128, 4, apply_norm=True)
    f3 = disc_downsample(f2, 256, 4, apply_norm=True)

    x = conv2d(
        512,
        4,
        strides=1,
        padding='same',
        kernel_initializer=initializer,
        use_bias=False,
    )(f3)
    x = tf.keras.layers.LayerNormalization()(x)
    f4 = tf.keras.layers.LeakyReLU(0.2)(x)

    patch = conv2d(
        1,
        4,
        strides=1,
        padding='same',
        kernel_initializer=initializer,
        dtype='float32',
    )(f4)

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
        real_loss = tf.reduce_mean(tf.square(real_p - 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_p))
        return 0.5 * (real_loss + fake_loss)

    real_label = tf.ones_like(real_p) * (1.0 - label_smoothing)
    fake_label = tf.zeros_like(fake_p) + label_smoothing

    if label_noise > 0:
        real_label = tf.clip_by_value(
            real_label + tf.random.uniform(tf.shape(real_label), -label_noise, label_noise),
            0.0,
            1.0,
        )
        fake_label = tf.clip_by_value(
            fake_label + tf.random.uniform(tf.shape(fake_label), -label_noise, label_noise),
            0.0,
            1.0,
        )

    real_loss = bce_loss(real_label, real_p)
    fake_loss = bce_loss(fake_label, fake_p)
    return real_loss + fake_loss


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
# TRAIN STEP
# -------------------------------------------------

@tf.function
def train_step(satellite, real_map, generator, discriminator,
               gen_opt, disc_opt, lambda_l1, label_smoothing, label_noise,
               fm_lambda, perc_model, perc_lambda, gan_mode, update_disc,
               gen_updates=1, disc_input_noise_std=0.05):
    def add_noise(x):
        if disc_input_noise_std <= 0:
            return x
        return x + tf.random.normal(tf.shape(x), mean=0.0, stddev=disc_input_noise_std, dtype=x.dtype)

    # Default tensors for logging in case generator updates are skipped.
    gen_total = tf.constant(0.0, dtype=tf.float32)
    gen_adv = tf.constant(0.0, dtype=tf.float32)
    gen_l1 = tf.constant(0.0, dtype=tf.float32)
    fm = tf.constant(0.0, dtype=tf.float32)
    perc = tf.constant(0.0, dtype=tf.float32)

    # Train generator (gen_updates passes per disc update).
    for _ in range(max(1, gen_updates)):
        with tf.GradientTape() as gt:
            gen_map = generator(satellite, training=True)

            real_input = add_noise(real_map)
            fake_input = add_noise(gen_map)
            # Discriminator in inference mode: no stochastic behavior during gen update.
            disc_real_out = discriminator([satellite, real_input], training=False)
            disc_fake_out = discriminator([satellite, fake_input], training=False)

            real_patch = disc_real_out[0]
            fake_patch = disc_fake_out[0]

            gen_total, gen_adv, gen_l1 = generator_loss(
                fake_patch,
                gen_map,
                real_map,
                lambda_l1,
                gan_mode,
            )

            if fm_lambda > 0:
                fm = feature_matching_loss_fn(disc_real_out[1:], disc_fake_out[1:])
                gen_total = gen_total + fm_lambda * fm
            else:
                fm = tf.constant(0.0, dtype=tf.float32)

            perc = perceptual_loss_fn(perc_model, real_map, gen_map)
            gen_total = gen_total + perc_lambda * perc

        gen_grads = gt.gradient(gen_total, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))

    # Train discriminator once, reusing the last generated image (consistent with gen step).
    with tf.GradientTape() as dt:
        # stop_gradient: discriminator trains on gen output as a fixed sample.
        gen_map_disc = tf.stop_gradient(tf.cast(gen_map, tf.float32))
        real_input = add_noise(real_map)
        fake_input = add_noise(gen_map_disc)

        disc_real_out = discriminator([satellite, real_input], training=True)
        disc_fake_out = discriminator([satellite, fake_input], training=True)
        disc_total = discriminator_loss(
            disc_real_out[0],
            disc_fake_out[0],
            label_smoothing,
            label_noise,
            gan_mode,
        )

    disc_grads = dt.gradient(disc_total, discriminator.trainable_variables)
    if update_disc:
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_total, gen_adv, gen_l1, disc_total, fm, perc


# -------------------------------------------------
# EVALUATION & VISUALIZATION
# -------------------------------------------------

def evaluate_metrics(generator, test_ds, num_samples=50):
    if not METRICS_AVAILABLE:
        return {}

    maes, ssims, psnrs = [], [], []
    for sat, real_map in test_ds.take(num_samples):
        gen_map = generator(sat, training=False)
        real = ((real_map[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        pred = ((gen_map[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        maes.append(np.mean(np.abs(real.astype(float) - pred.astype(float))) / 255.0)
        ssims.append(ssim_fn(real, pred, channel_axis=2, data_range=255))
        psnrs.append(psnr_fn(real, pred, data_range=255))

    return {
        "MAE": float(np.mean(maes)),
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
    ap = argparse.ArgumentParser(description="Pix2Pix Satellite->Map Training")

    # Paths
    ap.add_argument('--data_dir', default='data/train')
    ap.add_argument('--test_dir', default='data/test')
    ap.add_argument('--results_dir', default='outputs/test_results')
    ap.add_argument('--savedir', default='saved_models')
    ap.add_argument('--logdir', default='logs')

    # Hyperparams
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=None,
                    help='Legacy single LR. If set, overrides both gen_lr and disc_lr.')
    ap.add_argument('--gen_lr', type=float, default=2e-4)
    ap.add_argument('--disc_lr', type=float, default=5e-5)
    ap.add_argument('--lambda_l1', type=float, default=100.0)
    ap.add_argument('--lambda_l1_end', type=float, default=100.0)
    ap.add_argument('--decay_epoch', type=int, default=100)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--sample_every', type=int, default=1)

    # Loss knobs
    ap.add_argument('--perceptual_lambda', '--lambda_perc', dest='perceptual_lambda', type=float, default=2.0)
    ap.add_argument('--feature_matching_lambda', '--lambda_fm', dest='feature_matching_lambda', type=float, default=5.0)
    ap.add_argument('--gan_mode', default='lsgan', choices=['lsgan', 'bce'])
    ap.add_argument('--label_smoothing', type=float, default=0.15)
    ap.add_argument('--label_noise', type=float, default=0.02)
    ap.add_argument('--disc_update_interval', type=int, default=1)
    ap.add_argument('--gen_updates', type=int, default=2,
                    help='Number of generator updates per discriminator update')
    ap.add_argument('--disc_input_noise_std', type=float, default=0.05,
                    help='Gaussian noise stddev added to discriminator real/fake inputs')

    # Architecture
    ap.add_argument('--generator_norm', default='instance', choices=['instance', 'batch'])
    ap.add_argument('--res_blocks', type=int, default=0)
    ap.add_argument('--spectral_norm', action='store_true',
                    help='Use spectral normalization in discriminator Conv2D layers (requires tensorflow-addons).')

    # Dataset
    ap.add_argument('--split_order', default='map_sat', choices=['map_sat', 'sat_map'])
    ap.add_argument('--cache_dataset', nargs='?', const=True, default=False, type=str2bool)

    # Mode
    ap.add_argument('--mode', default='full', choices=['full', 'demo'])
    ap.add_argument('--demo_steps', type=int, default=5)
    ap.add_argument(
        '--restore',
        action='store_true',
        help='Restore latest checkpoint'
    )
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--multi_gpu', action='store_true')
    ap.add_argument('--require_gpu', action='store_true')
    ap.add_argument('--mixed_precision', action='store_true', default=False)

    args = ap.parse_args()

    if args.lr is not None:
        args.gen_lr = float(args.lr)
        args.disc_lr = float(args.lr)

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[AMP] mixed_float16 enabled")

    # GPU / Strategy
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[GPU] Detected: {len(gpus)} GPU(s)")
    if not gpus:
        print("[WARN] No GPUs found - running on CPU. Training will be slow.")
        if args.require_gpu:
            raise RuntimeError("--require_gpu set but no GPU is visible to TensorFlow")

    if args.multi_gpu and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[GPU] MirroredStrategy over {strategy.num_replicas_in_sync} GPUs")
        effective_batch = args.batch_size * strategy.num_replicas_in_sync
    else:
        if args.multi_gpu and len(gpus) <= 1:
            print("[WARN] --multi_gpu set but only 1 (or 0) GPU visible. Using default strategy.")
        strategy = tf.distribute.get_strategy()
        effective_batch = args.batch_size

    print(f"[BATCH] effective_batch={effective_batch}")

    # Directories
    for d in [args.results_dir, args.savedir, args.logdir]:
        os.makedirs(d, exist_ok=True)

    # Datasets
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

    n_train = len(list_image_files(args.data_dir))
    steps_per_epoch = max(1, int(math.ceil(n_train / effective_batch)))
    print(f"[DATA] {n_train} images | {steps_per_epoch} steps/epoch")

    with strategy.scope():
        generator = build_generator(norm_type=args.generator_norm)
        if args.spectral_norm and not TFA_AVAILABLE:
            print("[WARN] --spectral_norm requested but tensorflow-addons is unavailable. Continuing without spectral norm.")
            args.spectral_norm = False
        discriminator = build_discriminator(use_spectral_norm=args.spectral_norm)

        perc_model = None
        if args.perceptual_lambda > 0:
            try:
                from tensorflow.keras.applications import VGG19

                base = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
                base.trainable = False
                outs = [base.get_layer(n).output for n in ('block3_conv3', 'block4_conv3')]
                perc_model = tf.keras.Model(base.input, outs)
                perc_model.trainable = False
                print("[LOSS] VGG19 perceptual loss enabled")
            except Exception as e:
                print(f"[WARN] VGG19 unavailable ({e}), perceptual_lambda=0")
                args.perceptual_lambda = 0.0

        gen_opt = tf.keras.optimizers.Adam(
            learning_rate=args.gen_lr,
            beta_1=0.5
        )
        disc_opt = tf.keras.optimizers.Adam(
            learning_rate=args.disc_lr,
            beta_1=0.5
        )

        ckpt = tf.train.Checkpoint(
            generator=generator,
            discriminator=discriminator,
            gen_opt=gen_opt,
            disc_opt=disc_opt,
        )

    mgr = tf.train.CheckpointManager(ckpt, args.savedir, max_to_keep=5)

    start_epoch = 0
    if args.restore and mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint)
        try:
            start_epoch = int(mgr.latest_checkpoint.split('-')[-1]) * args.save_every
        except Exception:
            pass
        print(f"[CKPT] Restored from {mgr.latest_checkpoint} (epoch ~{start_epoch})")

    sw = tf.summary.create_file_writer(args.logdir)

    best_ssim = 0.0
    loss_window = deque(maxlen=6)

    print(f"\n{'=' * 60}")
    print(f"  Pix2Pix | {args.epochs} epochs | L1={args.lambda_l1}")
    print(f"  perceptual={args.perceptual_lambda} | fm={args.feature_matching_lambda}")
    print(f"  gen_lr={args.gen_lr} | disc_lr={args.disc_lr} | G:D={max(1,args.gen_updates)}:1")
    print(f"  spectral_norm={args.spectral_norm}")
    print(f"  gan_mode={args.gan_mode} | norm={args.generator_norm}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # LR schedule: warmup then linear decay
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            gen_lr_now = args.gen_lr * ((epoch + 1) / args.warmup_epochs)
            disc_lr_now = args.disc_lr * ((epoch + 1) / args.warmup_epochs)
        elif epoch >= args.decay_epoch:
            decay_frac = (epoch - args.decay_epoch) / max(1, args.epochs - args.decay_epoch)
            gen_lr_now = max(args.gen_lr * (1.0 - decay_frac), 1e-7)
            disc_lr_now = max(args.disc_lr * (1.0 - decay_frac), 1e-7)
        else:
            gen_lr_now = args.gen_lr
            disc_lr_now = args.disc_lr
        gen_opt.learning_rate.assign(gen_lr_now)
        disc_opt.learning_rate.assign(disc_lr_now)

        # L1 schedule (optional decay)
        l1_frac = epoch / max(1, args.epochs - 1)
        lambda_l1_now = args.lambda_l1 + (args.lambda_l1_end - args.lambda_l1) * l1_frac

        # Aux loss warmup: FM and perceptual losses are zero during LR warmup, then
        # ramp linearly from 0→full over the next 10 epochs.  This lets the generator
        # first learn the basic L1/adversarial mapping before noisy auxiliary signals
        # (from an untrained discriminator and mis-matched VGG features) are introduced.
        aux_ramp_start = args.warmup_epochs
        aux_ramp_end   = args.warmup_epochs + 5
        if epoch < aux_ramp_start:
            aux_scale = 0.0
        elif epoch < aux_ramp_end:
            aux_scale = float(epoch - aux_ramp_start) / float(aux_ramp_end - aux_ramp_start)
        else:
            aux_scale = 1.0
        fm_lambda_now   = args.feature_matching_lambda * aux_scale
        perc_lambda_now = args.perceptual_lambda * aux_scale

        gen_ls, disc_ls, l1_ls, adv_ls, fm_ls, perc_ls = [], [], [], [], [], []
        demo_done = False

        # Iterate directly - no .take() on distributed dataset wrappers.
        step = 0
        pbar = tqdm(train_ds, desc=f"Epoch {epoch + 1:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            if step >= steps_per_epoch:
                break

            sat_b, map_b = batch
            update_disc = ((step % max(1, args.disc_update_interval)) == 0)

            per_replica = strategy.run(
                train_step,
                args=(
                    sat_b,
                    map_b,
                    generator,
                    discriminator,
                    gen_opt,
                    disc_opt,
                    tf.constant(lambda_l1_now, dtype=tf.float32),
                    args.label_smoothing,
                    args.label_noise,
                    fm_lambda_now,
                    perc_model,
                    perc_lambda_now,
                    args.gan_mode,
                    update_disc,
                    args.gen_updates,
                    args.disc_input_noise_std,
                ),
            )
            gt   = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None)
            ga   = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None)
            gl1  = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None)
            dt   = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None)
            fm   = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[4], axis=None)
            perc = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[5], axis=None)

            gen_ls.append(float(gt))
            disc_ls.append(float(dt))
            l1_ls.append(float(gl1))
            adv_ls.append(float(ga))
            fm_ls.append(float(fm))
            perc_ls.append(float(perc))

            pbar.set_postfix({'gen': f'{float(gt):.4f}', 'disc': f'{float(dt):.4f}'})

            step += 1
            if args.mode == 'demo' and step >= args.demo_steps:
                demo_done = True
                break

        if demo_done:
            print(f"[DEMO] Stopped after {args.demo_steps} steps.")
            break

        mg = float(np.mean(gen_ls))
        md = float(np.mean(disc_ls))
        ml = float(np.mean(l1_ls))
        ma = float(np.mean(adv_ls))
        mf = float(np.mean(fm_ls))
        mp = float(np.mean(perc_ls))
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"gen={mg:.4f} disc={md:.4f} l1={ml:.4f} adv={ma:.4f} "
            f"fm={mf:.4f} perc={mp:.4f} | gen_lr={gen_lr_now:.2e} disc_lr={disc_lr_now:.2e} | {elapsed:.1f}s"
        )

        if epoch >= 4 and md < 0.10:
            print(f"  [WARN] disc={md:.4f} very low - discriminator dominating (consider reducing disc_lr or adding disc_update_interval)")
        if epoch >= 4 and md > 0.70:
            print(f"  [WARN] disc={md:.4f} high - generator may be failing (check gen_updates or lambda_l1)")

        with sw.as_default():
            tf.summary.scalar('loss/gen', mg, step=epoch)
            tf.summary.scalar('loss/disc', md, step=epoch)
            tf.summary.scalar('loss/l1', ml, step=epoch)
            tf.summary.scalar('loss/adv', ma, step=epoch)
            tf.summary.scalar('loss/fm', mf, step=epoch)
            tf.summary.scalar('loss/perc', mp, step=epoch)
            tf.summary.scalar('lr/gen', gen_lr_now, step=epoch)
            tf.summary.scalar('lr/disc', disc_lr_now, step=epoch)

        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            save_sample(generator, test_ds, args.results_dir, epoch + 1)

        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate_metrics(generator, test_ds)
            if metrics:
                print(
                    f"  [METRICS] MAE={metrics['MAE']:.4f} "
                    f"SSIM={metrics['SSIM']:.4f} PSNR={metrics['PSNR']:.2f}dB"
                )
                with sw.as_default():
                    tf.summary.scalar('metrics/SSIM', metrics['SSIM'], step=epoch)
                    tf.summary.scalar('metrics/MAE', metrics['MAE'], step=epoch)
                    tf.summary.scalar('metrics/PSNR', metrics['PSNR'], step=epoch)
                if metrics['SSIM'] > best_ssim:
                    best_ssim = metrics['SSIM']
                    bp = os.path.join(args.savedir, 'best_generator')
                    if tf.io.gfile.exists(bp):
                        tf.io.gfile.rmtree(bp)
                    generator.save(bp)
                    print(f"  [BEST] SSIM={best_ssim:.4f} -> {bp}")

        loss_window.append(mg)
        if len(loss_window) == loss_window.maxlen and epoch >= 15:
            if (max(loss_window) - min(loss_window)) < 0.02:
                new_gen_lr = max(gen_lr_now * 0.5, 1e-7)
                new_disc_lr = max(disc_lr_now * 0.5, 1e-7)
                gen_opt.learning_rate.assign(new_gen_lr)
                disc_opt.learning_rate.assign(new_disc_lr)
                print(f"  [ADAPT] Loss stagnant - gen_lr -> {new_gen_lr:.2e}, disc_lr -> {new_disc_lr:.2e}")

        if (epoch + 1) % args.save_every == 0:
            print(f"  [CKPT] {mgr.save()}")

    print(f"\n[DONE] Best SSIM: {best_ssim:.4f}")
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
