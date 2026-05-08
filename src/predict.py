"""
Prediction helpers for trained skin lesion classification models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from .config import BATCH_SIZE, IMG_SIZE
    from .preprocessing import load_for_efficientnet
except ImportError:
    from config import BATCH_SIZE, IMG_SIZE
    from preprocessing import load_for_efficientnet


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    """
    Load a saved Keras model.
    """
    return tf.keras.models.load_model(model_path)


def preprocess_single_image(image_path: str | Path, image_size: int = IMG_SIZE) -> tf.Tensor:
    """
    Load one image and add a batch dimension for model prediction.
    """
    image = load_for_efficientnet(tf.constant(str(image_path)), image_size=image_size)
    return tf.expand_dims(image, axis=0)


def predict_topk(
    image_path: str | Path,
    model: tf.keras.Model,
    class_names: list[str],
    k: int = 3,
) -> list[dict[str, float | str]]:
    """
    Return the top-k predicted lesion classes for one image.
    """
    image = preprocess_single_image(image_path)
    probabilities = model.predict(image, verbose=0)[0]

    top_indices = np.argsort(probabilities)[-k:][::-1]

    return [
        {
            "class_name": class_names[index],
            "probability": float(probabilities[index]),
        }
        for index in top_indices
    ]


def predict_malignancy(
    image_path: str | Path,
    model: tf.keras.Model,
    threshold: float = 0.5,
) -> dict[str, float | str]:
    """
    Predict malignant-vs-benign risk for one image.

    Returns a dictionary with the malignant probability and thresholded label.
    """
    image = preprocess_single_image(image_path)
    probability = float(model.predict(image, verbose=0).ravel()[0])
    label = "malignant" if probability >= threshold else "benign"

    return {
        "malignant_probability": probability,
        "threshold": float(threshold),
        "predicted_label": label,
    }


def predict_malignancy_batch(
    image_paths: list[str | Path],
    model: tf.keras.Model,
    threshold: float = 0.5,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, float | str]]:
    """
    Predict malignant-vs-benign risk for several images in one batched pass.
    """
    if not image_paths:
        return []

    path_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])
    image_ds = (
        path_ds
        .map(
            lambda p: load_for_efficientnet(p, image_size=IMG_SIZE),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    probabilities = model.predict(image_ds, verbose=0).ravel()

    return [
        {
            "image_path": str(path),
            "malignant_probability": float(prob),
            "threshold": float(threshold),
            "predicted_label": "malignant" if prob >= threshold else "benign",
        }
        for path, prob in zip(image_paths, probabilities)
    ]
