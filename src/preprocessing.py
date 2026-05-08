"""
Image preprocessing utilities for skin lesion classification.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .config import IMG_SIZE
except ImportError:
    from config import IMG_SIZE


def load_image_cv2(
    image_path: str | Path,
    image_size: int | tuple[int, int] = IMG_SIZE,
    scale_to_unit_range: bool = True,
) -> np.ndarray:
    """
    Load an image with OpenCV, convert BGR to RGB, resize, and optionally scale.

    Args:
        image_path: Path to the image file.
        image_size: Integer size or `(height, width)` tuple.
        scale_to_unit_range: Whether to scale pixel values to `[0, 1]`.

    Returns:
        RGB image as a NumPy array.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # cv2.resize expects (width, height); image_size is (height, width) or a square int.
    if isinstance(image_size, int):
        cv2_size = (image_size, image_size)
    else:
        height, width = image_size
        cv2_size = (width, height)

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, cv2_size, interpolation=cv2.INTER_AREA)

    if scale_to_unit_range:
        image = image.astype("float32") / 255.0

    return image


def random_augment_numpy(image: np.ndarray) -> np.ndarray:
    """
    Lightweight augmentation for non-TF workflows (e.g. quick experiments).

    Not called in the standard training pipeline — make_tf_dataset uses
    make_keras_augmentation instead. Available via preprocess_image(path, augment=True).
    """
    out = image.copy()

    if np.random.rand() < 0.5:
        out = np.fliplr(out)

    if np.random.rand() < 0.5:
        out = np.flipud(out)

    if np.random.rand() < 0.3:
        factor = np.random.uniform(0.85, 1.15)
        out = np.clip(out * factor, 0, 1)

    return out.astype("float32")


def preprocess_image(
    image_path: str | Path,
    image_size: int | tuple[int, int] = IMG_SIZE,
    augment: bool = False,
) -> np.ndarray:
    """
    Load and preprocess one image for classical ML or visualisation workflows.
    """
    image = load_image_cv2(image_path, image_size=image_size, scale_to_unit_range=True)
    if augment:
        image = random_augment_numpy(image)
    return image


def make_keras_augmentation(seed: int = 42):
    """
    Build the Keras augmentation pipeline used before EfficientNetB0.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, seed=seed),
            tf.keras.layers.RandomZoom(0.05, 0.05, seed=seed),
            tf.keras.layers.RandomContrast(0.05, seed=seed),
        ],
        name="augmentation",
    )


def load_for_efficientnet(path, image_size: int = IMG_SIZE):
    """
    Read an image path tensor and preprocess it for EfficientNetB0.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32)
    return preprocess_input(image)


def make_tf_dataset(
    paths,
    labels,
    batch_size: int = 32,
    image_size: int = IMG_SIZE,
    training: bool = False,
    one_hot_classes: int | None = None,
    seed: int = 42,
):
    """
    Build a `tf.data.Dataset` from image paths and labels.
    """
    import tensorflow as tf

    augmentation = make_keras_augmentation(seed=seed) if training else None

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))

    def _parse(path, label):
        image = load_for_efficientnet(path, image_size=image_size)
        if augmentation is not None:
            image = augmentation(image, training=True)
        if one_hot_classes is not None:
            label_out = tf.one_hot(tf.cast(label, tf.int32), one_hot_classes)
        else:
            label_out = tf.cast(label, tf.float32)
        return image, label_out

    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 2048), seed=seed)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def rgb_hist_features(image_paths, bins: int = 16) -> np.ndarray:
    """
    Extract simple RGB histogram features for baseline scikit-learn models.
    """
    features = []

    for path in image_paths:
        image = preprocess_image(path)
        red = np.histogram(image[..., 0], bins=bins, range=(0, 1), density=True)[0]
        green = np.histogram(image[..., 1], bins=bins, range=(0, 1), density=True)[0]
        blue = np.histogram(image[..., 2], bins=bins, range=(0, 1), density=True)[0]
        features.append(np.concatenate([red, green, blue]))

    return np.asarray(features, dtype="float32")
