"""
Preprocessing utilities for skin lesion image classification.

This module can be expanded once the project dataset and model pipeline
are added to the repository.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_image(image_path: str | Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image, resize it, and return it as a NumPy array.

    Args:
        image_path: Path to the image file.
        target_size: Desired image size as (width, height).

    Returns:
        NumPy array representation of the resized RGB image.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    image = image.resize(target_size)

    return np.array(image)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to the range [0, 1].

    Args:
        image: Image array with pixel values from 0 to 255.

    Returns:
        Normalized image array.
    """
    return image.astype("float32") / 255.0
