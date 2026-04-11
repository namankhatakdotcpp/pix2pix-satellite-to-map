import tensorflow as tf
import os

IMG_HEIGHT = 256
IMG_WIDTH = 256

def load(image_file):
    """
    Loads an image file and splits it into input (satellite) and target (map).
    The dataset contains images concatenated side by side: [map | satellite].
    """

    # Read and decode image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split into input and target (each half is 256x256)
    w = tf.shape(image)[1] // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    # Convert to float32 and normalize [-1, 1]
    input_image = tf.cast(input_image, tf.float32) / 127.5 - 1
    real_image = tf.cast(real_image, tf.float32) / 127.5 - 1

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    # Data augmentation: random jitter and flip
    input_image, real_image = random_jitter(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    real_image = tf.image.resize(real_image, [IMG_HEIGHT, IMG_WIDTH])
    return input_image, real_image


def random_jitter(input_image, real_image):
    # Resize to 286x286 and crop back to 256x256 (data augmentation)
    input_image = tf.image.resize(input_image, [286, 286])
    real_image = tf.image.resize(real_image, [286, 286])

    stacked = tf.stack([input_image, real_image], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    input_image, real_image = cropped[0], cropped[1]

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def create_dataset(data_dir, batch_size=1, is_train=True):
    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*.jpg'))
    if is_train:
        dataset = dataset.map(load_image_train,
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(100)
    else:
        dataset = dataset.map(load_image_test,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    return dataset
