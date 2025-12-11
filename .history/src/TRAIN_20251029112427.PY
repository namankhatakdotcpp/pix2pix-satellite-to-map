#!/usr/bin/env python3
"""
train.py

Train Pix2Pix (U-Net generator + PatchGAN discriminator) using:
 - src/load_data.py
 - src/pix2pix_model.py

Supports:
  --mode demo   : small demo run using data/ (fast)
  --mode full   : full run using data_dir (slow, use full dataset path)

Example (demo):
  python src/train.py --mode demo --data_dir ../data/train --results_dir ../outputs --savedir ../saved_models --epochs 5 --batch_size 2

Example (full):
  python src/train.py --mode full --data_dir /path/to/full/maps/train --results_dir ../outputs --savedir ../saved_models --epochs 50 --batch_size 1
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# local imports (make sure PYTHONPATH includes project root or run from project root)
from src.load_data import create_dataset, IMG_HEIGHT, IMG_WIDTH
from src.pix2pix_model import build_generator, build_discriminator, discriminator_loss, generator_loss, get_optimizers


# Utility helpers

def makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_sample_image(out_dir, epoch, src_img, tar_img, gen_img, idx=0):
    """
    Save a single triplet (input, target, generated) as PNG in out_dir.
    src_img, tar_img, gen_img are numpy arrays in [-1,1].
    """
    makedirs(out_dir)
    # convert to 0..1
    s = (src_img[idx] + 1) / 2.0
    t = (tar_img[idx] + 1) / 2.0
    g = (gen_img[idx] + 1) / 2.0

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(np.clip(s, 0.0, 1.0)); axs[0].set_title('Input'); axs[0].axis('off')
    axs[1].imshow(np.clip(t, 0.0, 1.0)); axs[1].set_title('Target'); axs[1].axis('off')
    axs[2].imshow(np.clip(g, 0.0, 1.0)); axs[2].set_title('Generated'); axs[2].axis('off')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"epoch_{epoch+1:03d}_sample_{idx+1}.png")
    fig.savefig(out_path)
    plt.close(fig)


# Training step function

@tf.function
def train_step(input_image, target, generator, discriminator, gen_optimizer, disc_optimizer, LAMBDA):
    """
    Performs one training step:
     - generates fake image
     - computes losses for discriminator and generator
     - applies gradients
    Returns: dict with losses
    """
    with tf.GradientTape(persistent=True) as tape:
        gen_output = generator(input_image, training=True)

        disc_real = discriminator([input_image, target], training=True)
        disc_generated = discriminator([input_image, gen_output], training=True)

        # compute losses
        disc_loss = discriminator_loss(disc_real, disc_generated)
        gen_total_loss, gen_adv_loss, gen_l1_loss = generator_loss(disc_generated, gen_output, target, LAMBDA=LAMBDA)

    # compute gradients
    gen_gradients = tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)

    # apply gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return {
        "gen_total": gen_total_loss,
        "gen_adv": gen_adv_loss,
        "gen_l1": gen_l1_loss,
        "disc": disc_loss
    }


# Training loop

def train(args):
    print("Starting training with args:", args)

    # prepare directories
    makedirs(args.results_dir)
    makedirs(args.savedir)

    # create dataset(s)
    is_demo = (args.mode == "demo")
    # create training dataset
    train_ds = create_dataset(args.data_dir, batch_size=args.batch_size, is_train=True)
    # create test dataset (for visualization) - we will take one batch
    test_ds = create_dataset(args.data_dir, batch_size=args.batch_size, is_train=False)

    # build models
    generator = build_generator(image_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    discriminator = build_discriminator(image_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # optimizers
    gen_opt, disc_opt = get_optimizers(lr=args.lr, beta_1=args.beta1)

    # training hyperparams
    LAMBDA = args.lmbda

    # checkpoints using tf.train.Checkpoint
    checkpoint_prefix = os.path.join(args.savedir, "ckpt")
    ckpt = tf.train.Checkpoint(generator=generator,
                               discriminator=discriminator,
                               gen_optimizer=gen_opt,
                               disc_optimizer=disc_opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.savedir, max_to_keep=5)

    # restore if requested
    if args.restore and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from", ckpt_manager.latest_checkpoint)
    elif args.restore:
        print("No checkpoint found to restore.")

    # sample batch for visualization
    sample_input = None
    sample_target = None
    for sample in test_ds.take(1):
        sample_input, sample_target = sample[0].numpy(), sample[1].numpy()
        break
    if sample_input is None:
        raise RuntimeError("Unable to fetch a sample batch from test dataset for visualization.")

    # training loop
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else tf.data.experimental.cardinality(train_ds).numpy()
    if steps_per_epoch == -2:  # cardinality unknown
        steps_per_epoch = 100

    print(f"Training: epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, batch_size={args.batch_size}")

    global_step = 0
    for epoch in range(args.epochs):
        start = time.time()
        step = 0
        for batch in train_ds.take(steps_per_epoch):
            step += 1
            global_step += 1
            input_image, target = batch[0], batch[1]  # tf.Tensor
            losses = train_step(input_image, target, generator, discriminator, gen_opt, disc_opt, LAMBDA)

            # logging
            if step % args.log_every == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Step {step}/{steps_per_epoch} "
                      f"gen_total={losses['gen_total']:.4f} gen_l1={losses['gen_l1']:.4f} disc={losses['disc']:.4f}")

            # quick demo break for faster runs if requested
            if is_demo and global_step >= args.demo_max_steps:
                print("Demo max steps reached, finishing epoch early.")
                break

        # end of epoch: save sample outputs and checkpoint if needed
        gen_out = generator(sample_input, training=False).numpy()
        save_sample_image(args.results_dir, epoch, sample_input, sample_target, gen_out, idx=0)
        # save checkpoint every epoch (configurable)
        if (epoch + 1) % args.save_every == 0:
            ckpt_manager.save()
            # also save a copy of generator weights
            gen_path = os.path.join(args.savedir, f"generator_epoch_{epoch+1:03d}.h5")
            generator.save(gen_path)
            print(f"Saved checkpoint and generator to {args.savedir}")

        print(f"Epoch {epoch+1} completed in {time.time()-start:.1f}s")

    print("Training finished.")


# CLI / Arg parsing

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["demo", "full"], default="demo", help="demo (small) or full (large) mode")
    p.add_argument("--data_dir", type=str, default="../data/train", help="directory containing concatenated images for demo (relative to src/)")
    p.add_argument("--results_dir", type=str, default="../outputs", help="where to save sample outputs (relative to src/)")
    p.add_argument("--savedir", type=str, default="../saved_models", help="checkpoint / model save directory (relative to src/)")
    p.add_argument("--epochs", type=int, default=10, help="number of epochs")
    p.add_argument("--batch_size", type=int, default=2, help="batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--beta1", type=float, default=0.5, help="adam beta1")
    p.add_argument("--lmbda", type=float, default=100.0, help="L1 loss weight")
    p.add_argument("--save_every", type=int, default=1, help="save checkpoint every N epochs")
    p.add_argument("--log_every", type=int, default=10, help="log progress every N steps")
    p.add_argument("--demo_max_steps", type=int, default=200, help="max global steps in demo mode (keeps run short)")
    p.add_argument("--steps_per_epoch", type=int, default=0, help="limit steps per epoch (0 = use full epoch)")
    p.add_argument("--restore", action="store_true", help="restore from latest checkpoint if available")
    return p.parse_args()


# Main

if __name__ == "__main__":
    args = parse_args()
    # Ensure relative paths are interpreted from repo root (recommended to run from repo root)
    # If you're running from src/ directly, adjust paths accordingly.
    # Example usage from repo root: python src/train.py ...
    train(args)
