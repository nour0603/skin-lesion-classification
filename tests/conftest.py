import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def tmp_jpeg(tmp_path):
    """Small random JPEG for image-loading tests."""
    path = tmp_path / "test_image.jpg"
    arr = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return path
