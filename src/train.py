"""
train.py — Enhanced Pix2Pix Training Script
Fixes applied:
  1. Default paths are repo-relative (no more ../)
  2. Tensor float casting in all logging
  3. Demo early-stop breaks outer loop too
Enhancements:
  - Multi-GPU support via MirroredStrategy (8x A6000 ready)
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
import numpy as np
import tensorflow as tf
import imageio
from pathlib import Path
from tqdm import tqdm

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

def load_image(image_path):
    """Load and split a paired satellite|map image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    w = tf.shape(img)[1]
    w = w // 2
    # NOTE: In the maps dataset, LEFT half = map, RIGHT half = satellite
    # Input to generator = satellite, Target = map
    real_map = img[:, :w, :]       # left = map (ground truth)
    satellite = img[:, w:, :]      # right = satellite (input)
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


def load_train_image(image_path):
    satellite, real_map = load_image(image_path)
    satellite, real_map = random_jitter(satellite, real_map)
    satellite, real_map = normalize(satellite, real_map)
    return satellite, real_map


def load_test_image(image_path):
    satellite, real_map = load_image(image_path)
    satellite = tf.image.resize(satellite, [256, 256])
    real_map  = tf.image.resize(real_map,  [256, 256])
    satellite, real_map = normalize(satellite, real_map)
    return satellite, real_map


def build_dataset(data_dir, batch_size, is_train=True, buffer_size=400):
    pattern = os.path.join(data_dir, "*.jpg")
    files = tf.data.Dataset.list_files(pattern)
    if is_train:
        ds = files.map(load_train_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = files.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def downsample(filters, size, apply_batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=init, use_bias=False))
    if apply_batchnorm:
        block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.LeakyReLU())
    return block


def upsample(filters, size, apply_dropout=False):
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=init, use_bias=False))
    block.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block


def build_generator():
    """U-Net generator with skip connections."""
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    init = tf.random_normal_initializer(0., 0.02)

    # Encoder (downsampling)
    down_stack = [
        downsample(64,  4, apply_batchnorm=False),  # (128, 128, 64)
        downsample(128, 4),                          # (64, 64, 128)
        downsample(256, 4),                          # (32, 32, 256)
        downsample(512, 4),                          # (16, 16, 512)
        downsample(512, 4),                          # (8, 8, 512)
        downsample(512, 4),                          # (4, 4, 512)
        downsample(512, 4),                          # (2, 2, 512)
        downsample(512, 4),                          # (1, 1, 512)
    ]
    # Decoder (upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True),   # (2, 2, 1024)
        upsample(512, 4, apply_dropout=True),   # (4, 4, 1024)
        upsample(512, 4, apply_dropout=True),   # (8, 8, 1024)
        upsample(512, 4),                       # (16, 16, 1024)
        upsample(256, 4),                       # (32, 32, 512)
        upsample(128, 4),                       # (64, 64, 256)
        upsample(64,  4),                       # (128, 128, 128)
    ]
    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=init, activation='tanh')  # (256, 256, 3)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

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
    x = tf.keras.layers.Concatenate()([inp, target])  # (256, 256, 6)
    x = downsample(64,  4, apply_batchnorm=False)(x)  # (128, 128, 64)
    x = downsample(128, 4)(x)                          # (64, 64, 128)
    x = downsample(256, 4)(x)                          # (32, 32, 256)
    x = tf.keras.layers.ZeroPadding2D()(x)             # (34, 34, 256)
    x = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=init, use_bias=False)(x)  # (31, 31, 512)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=init)(x)  # (30, 30, 1)
    return tf.keras.Model(inputs=[inp, target], outputs=x)


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100):
    """Adversarial + L1 loss."""
    adv_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss  = tf.reduce_mean(tf.abs(target - gen_output))
    total    = adv_loss + (lambda_l1 * l1_loss)
    return total, adv_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """BCE on real + fake."""
    real_loss = loss_object(tf.ones_like(disc_real_output),  disc_real_output)
    fake_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss


# ─────────────────────────────────────────────
# TRAIN STEP
# ─────────────────────────────────────────────

def train_step(satellite, real_map, generator, discriminator,
               gen_optimizer, disc_optimizer, lambda_l1):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_map = generator(satellite, training=True)

        disc_real = discriminator([satellite, real_map], training=True)
        disc_fake = discriminator([satellite, gen_map],  training=True)

        gen_total, gen_adv, gen_l1 = generator_loss(disc_fake, gen_map, real_map, lambda_l1)
        disc_total = discriminator_loss(disc_real, disc_fake)

    gen_grads  = gen_tape.gradient(gen_total,  generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_total, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads,  generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_total, gen_adv, gen_l1, disc_total


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
        return path


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
    parser = argparse.ArgumentParser(description="Pix2Pix Satellite→Map Training")
    # Paths — FIX 1: repo-relative defaults (no more ../)
    parser.add_argument('--data_dir',    default='data/train',           help='Training images dir')
    parser.add_argument('--test_dir',    default='data/test',            help='Test images dir')
    parser.add_argument('--results_dir', default='outputs/test_results', help='Sample output dir')
    parser.add_argument('--savedir',     default='saved_models',         help='Checkpoint dir')
    parser.add_argument('--logdir',      default='logs',                 help='TensorBoard log dir')
    # Training hyperparams
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=1)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--lambda_l1',   type=float, default=100.0,
                        help='L1 loss weight. Try 50 if output is blurry.')
    parser.add_argument('--decay_epoch', type=int,   default=100,
                        help='Epoch to start LR linear decay (paper default: 100)')
    parser.add_argument('--save_every',  type=int,   default=10)
    parser.add_argument('--eval_every',  type=int,   default=10)
    # Mode
    parser.add_argument('--mode',        default='full', choices=['full', 'demo'])
    parser.add_argument('--demo_steps',  type=int,   default=5)
    parser.add_argument('--restore',     action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--export',      action='store_true', help='Export SavedModel after training')
    # Multi-GPU
    parser.add_argument('--multi_gpu',   action='store_true',
                        help='Enable MirroredStrategy for multi-GPU training')
    args = parser.parse_args()

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

    # ── Build datasets ─────────────────────────────────────────────────
    train_ds = build_dataset(args.data_dir, effective_batch, is_train=True)
    test_ds  = build_dataset(args.test_dir, 1, is_train=False)

    # Count training samples
    n_train = len(tf.io.gfile.glob(os.path.join(args.data_dir, "*.jpg")))
    if n_train == 0:
        raise RuntimeError(f"No training images found in {args.data_dir} (expected *.jpg)")
    steps_per_epoch = max(1, int(math.ceil(n_train / effective_batch)))
    print(f"[DATA] {n_train} training images | {steps_per_epoch} steps/epoch")

    # ── Build models inside strategy scope ────────────────────────────
    with strategy.scope():
        generator     = build_generator()
        discriminator = build_discriminator()

        # Optimizers with LR variable (for decay schedule)
        gen_lr  = tf.Variable(args.lr, trainable=False, dtype=tf.float32)
        disc_lr = tf.Variable(args.lr, trainable=False, dtype=tf.float32)
        gen_optimizer  = tf.keras.optimizers.Adam(gen_lr,  beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=0.5)

    @tf.function
    def single_train_step(sat_batch, map_batch):
        return train_step(
            sat_batch,
            map_batch,
            generator,
            discriminator,
            gen_optimizer,
            disc_optimizer,
            args.lambda_l1,
        )

    if args.multi_gpu:
        distributed_train_ds = strategy.experimental_distribute_dataset(train_ds)

        @tf.function
        def distributed_train_step(dist_batch):
            def replica_step(sat_batch, map_batch):
                return train_step(
                    sat_batch,
                    map_batch,
                    generator,
                    discriminator,
                    gen_optimizer,
                    disc_optimizer,
                    args.lambda_l1,
                )

            per_replica = strategy.run(replica_step, args=dist_batch)
            gen_total = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None)
            gen_adv = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None)
            gen_l1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None)
            disc_total = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None)
            return gen_total, gen_adv, gen_l1, disc_total
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

    print(f"\n{'='*60}")
    print(f"  Pix2Pix Training — {args.epochs} epochs")
    print(f"  L1 weight: {args.lambda_l1} | LR: {args.lr}")
    print(f"  LR decay starts at epoch: {args.decay_epoch}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── LR linear decay (paper-faithful) ──────────────────────────
        if epoch >= args.decay_epoch:
            decay_steps = args.epochs - args.decay_epoch
            new_lr = args.lr * (1.0 - (epoch - args.decay_epoch) / max(1, decay_steps))
            new_lr = max(new_lr, 1e-7)
            gen_lr.assign(new_lr)
            disc_lr.assign(new_lr)

        # ── Batch loop ────────────────────────────────────────────────
        gen_losses, disc_losses = [], []
        demo_done = False  # FIX 3: flag to break outer loop in demo mode

        pbar = tqdm(enumerate(distributed_train_ds.take(steps_per_epoch)), total=steps_per_epoch,
                    desc=f"Epoch {epoch+1:03d}/{args.epochs}", leave=False)

        for step, batch in pbar:
            if args.multi_gpu:
                gen_total, gen_adv, gen_l1, disc_total = distributed_train_step(batch)
            else:
                sat_batch, map_batch = batch
                gen_total, gen_adv, gen_l1, disc_total = single_train_step(sat_batch, map_batch)
            # FIX 2: cast tensors to float before string formatting
            gen_losses.append(float(gen_total))
            disc_losses.append(float(disc_total))

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

        # ── Epoch-level stats ─────────────────────────────────────────
        mean_gen  = float(np.mean(gen_losses))
        mean_disc = float(np.mean(disc_losses))
        elapsed   = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"gen={mean_gen:.4f} disc={mean_disc:.4f} | "
              f"lr={float(gen_lr):.2e} | {elapsed:.1f}s")

        # ── TensorBoard ───────────────────────────────────────────────
        with summary_writer.as_default():
            tf.summary.scalar('loss/generator',     mean_gen,  step=epoch)
            tf.summary.scalar('loss/discriminator', mean_disc, step=epoch)
            tf.summary.scalar('lr/generator',       float(gen_lr), step=epoch)

        # ── Save sample image ─────────────────────────────────────────
        frame_path = save_sample(generator, test_ds, args.results_dir, epoch + 1)
        if frame_path:
            gif_frames.append(frame_path)

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