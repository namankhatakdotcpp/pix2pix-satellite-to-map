"""
pix2pix_model.py
----------------
Pix2Pix model components (TensorFlow / Keras):

- build_generator(image_shape=(256,256,3)) -> Keras Model (U-Net)
- build_discriminator(image_shape=(256,256,3)) -> Keras Model (PatchGAN)
- generator_loss(disc_generated_output, gen_output, target, LAMBDA=100)
- discriminator_loss(disc_real_output, disc_generated_output)
- get_optimizers(lr=2e-4, beta_1=0.5)
- save_model(model, path) / load_model(path)

Intended to be imported and used from train.py
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal
import os

# -----------------------------
# Generator (U-Net style)
# -----------------------------
def conv_block(x, filters, batchnorm=True):
    init = RandomNormal(0., 0.02)
    x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=init, use_bias=not batchnorm)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def deconv_block(x, skip, filters, dropout=False):
    init = RandomNormal(0., 0.02)
    x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same',
                               kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip])
    return x

def build_generator(image_shape=(256, 256, 3)):
    """
    Returns a U-Net generator model that maps input_image -> generated_image.
    Input and output shape: (256,256,3)
    """
    init = RandomNormal(0., 0.02)
    inputs = layers.Input(shape=image_shape)

    # Encoder
    e1 = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init)(inputs)  # no batchnorm on first
    e1 = layers.LeakyReLU(0.2)(e1)                          # 128x128
    e2 = conv_block(e1, 128)                               # 64x64
    e3 = conv_block(e2, 256)                               # 32x32
    e4 = conv_block(e3, 512)                               # 16x16
    e5 = conv_block(e4, 512)                               # 8x8
    e6 = conv_block(e5, 512)                               # 4x4
    e7 = conv_block(e6, 512)                               # 2x2

    # bottleneck
    b = layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init)(e7)  # 1x1
    b = layers.Activation('relu')(b)

    # Decoder (skip connections)
    d1 = deconv_block(b, e7, 512, dropout=True)  # 2x2
    d2 = deconv_block(d1, e6, 512, dropout=True) # 4x4
    d3 = deconv_block(d2, e5, 512, dropout=True) # 8x8
    d4 = deconv_block(d3, e4, 512, dropout=False)# 16x16
    d5 = deconv_block(d4, e3, 256, dropout=False)# 32x32
    d6 = deconv_block(d5, e2, 128, dropout=False)# 64x64
    d7 = deconv_block(d6, e1, 64, dropout=False) # 128x128

    # final upsample to original size
    init_out = RandomNormal(0., 0.02)
    out = layers.Conv2DTranspose(image_shape[2], 4, strides=2, padding='same',
                                 kernel_initializer=init_out, activation='tanh')(d7)  # 256x256

    model = Model(inputs=inputs, outputs=out, name='generator_unet')
    return model

# -----------------------------
# Discriminator (PatchGAN)
# -----------------------------
def build_discriminator(image_shape=(256,256,3)):
    """
    PatchGAN discriminator that takes (input, target) concatenated and outputs patch logits.
    Output shape will be (None, patch_h, patch_w, 1)
    """
    init = RandomNormal(0., 0.02)
    in_src = layers.Input(shape=image_shape, name='input_image')
    in_target = layers.Input(shape=image_shape, name='target_image')
    merged = layers.Concatenate()([in_src, in_target])  # channel-wise concat -> shape (256,256,6)

    d = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init)(merged)
    d = layers.LeakyReLU(0.2)(d)                 # 128

    d = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init, use_bias=False)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(0.2)(d)                 # 64

    d = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=init, use_bias=False)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(0.2)(d)                 # 32

    d = layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init, use_bias=False)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(0.2)(d)                 # 16

    patch_out = layers.Conv2D(1, 4, padding='same', kernel_initializer=init, activation='sigmoid')(d)  # patch output
    model = Model(inputs=[in_src, in_target], outputs=patch_out, name='discriminator_patch')
    return model

# -----------------------------
# Loss functions
# -----------------------------
# L1 loss weight (lambda) used by original pix2pix
DEFAULT_LAMBDA = 100.0

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Standard pix2pix discriminator loss:
    - real images labelled 1, generated labelled 0.
    disc_* are outputs of the PatchGAN (sigmoid), shape [batch, h, w, 1]
    """
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = (real_loss + generated_loss) * 0.5  # authors used loss_weights=[0.5] sometimes
    return total_disc_loss

def generator_adversarial_loss(disc_generated_output):
    """Adversarial part of generator loss (make discriminator predict 1 for generated)"""
    return bce(tf.ones_like(disc_generated_output), disc_generated_output)

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=DEFAULT_LAMBDA):
    """
    Full generator loss = adversarial loss + L1 loss (MAE) * lambda
    """
    adv_loss = generator_adversarial_loss(disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = adv_loss + (LAMBDA * l1_loss)
    return total_gen_loss, adv_loss, l1_loss

# -----------------------------
# Optimizers & helpers
# -----------------------------
def get_optimizers(lr=2e-4, beta_1=0.5):
    """Return (gen_optimizer, disc_optimizer)"""
    gen_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
    return gen_opt, disc_opt

def save_model(model, path):
    """Save a model (Keras .h5 or TF SavedModel depending on extension)"""
    makedirs_for_file(path)
    model.save(path)
    return path

def load_model(path, compile=False):
    """Load a saved Keras model"""
    return tf.keras.models.load_model(path, compile=compile)

def makedirs_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -----------------------------
# Quick sanity check (if run directly)
# -----------------------------
if __name__ == "__main__":
    gen = build_generator()
    disc = build_discriminator()
    print("Generator params:", gen.count_params())
    print("Discriminator params:", disc.count_params())
    # quick forward pass with random data
    x = tf.random.normal((1,256,256,3))
    y = gen(x)
    print("Generated shape:", y.shape)
    p = disc([x, y])
    print("Patch output shape:", p.shape)
