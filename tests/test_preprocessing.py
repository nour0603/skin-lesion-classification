import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    load_image_cv2,
    make_keras_augmentation,
    random_augment_numpy,
    rgb_hist_features,
)


def _write_jpeg(path, size=(64, 64), seed=0):
    arr = (np.random.default_rng(seed).random((*size, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return path


class TestLoadImageCv2:
    def test_output_shape(self, tmp_jpeg):
        image = load_image_cv2(tmp_jpeg, image_size=64)
        assert image.shape == (64, 64, 3)

    def test_scaled_values_in_unit_range(self, tmp_jpeg):
        image = load_image_cv2(tmp_jpeg, image_size=64, scale_to_unit_range=True)
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_unscaled_values_above_one(self, tmp_jpeg):
        image = load_image_cv2(tmp_jpeg, image_size=64, scale_to_unit_range=False)
        assert image.max() > 1.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image_cv2("/nonexistent/path/image.jpg")

    def test_tuple_size_respected(self, tmp_jpeg):
        image = load_image_cv2(tmp_jpeg, image_size=(32, 48))
        assert image.shape == (32, 48, 3)


class TestRandomAugmentNumpy:
    def test_preserves_shape(self):
        image = np.random.default_rng(0).random((64, 64, 3)).astype("float32")
        result = random_augment_numpy(image)
        assert result.shape == image.shape

    def test_output_clipped_to_unit_range(self):
        image = np.ones((64, 64, 3), dtype="float32") * 0.9
        result = random_augment_numpy(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_returns_float32(self):
        image = np.random.default_rng(0).random((64, 64, 3)).astype("float32")
        result = random_augment_numpy(image)
        assert result.dtype == np.float32

    def test_does_not_mutate_input(self):
        image = np.random.default_rng(0).random((64, 64, 3)).astype("float32")
        original = image.copy()
        random_augment_numpy(image)
        np.testing.assert_array_equal(image, original)


class TestRgbHistFeatures:
    def test_feature_shape(self, tmp_path):
        paths = [_write_jpeg(tmp_path / f"img_{i}.jpg", seed=i) for i in range(4)]
        features = rgb_hist_features(paths, bins=16)
        assert features.shape == (4, 48)  # 3 channels × 16 bins

    def test_output_dtype(self, tmp_path):
        path = _write_jpeg(tmp_path / "img.jpg")
        features = rgb_hist_features([path], bins=8)
        assert features.dtype == np.float32

    def test_single_image_returns_2d(self, tmp_path):
        path = _write_jpeg(tmp_path / "img.jpg")
        features = rgb_hist_features([path])
        assert features.ndim == 2
        assert features.shape[0] == 1


class TestMakeKerasAugmentation:
    @pytest.fixture(autouse=True)
    def require_tf(self):
        pytest.importorskip("tensorflow")

    def test_returns_sequential(self):
        import tensorflow as tf

        aug = make_keras_augmentation()
        assert isinstance(aug, tf.keras.Sequential)

    def test_applies_to_image_tensor(self):
        import tensorflow as tf

        aug = make_keras_augmentation(seed=0)
        image = tf.ones((380, 380, 3), dtype=tf.float32)
        result = aug(image, training=True)
        assert result.shape == (380, 380, 3)

    def test_inference_mode_is_deterministic(self):
        import tensorflow as tf

        aug = make_keras_augmentation(seed=0)
        image = tf.constant(
            np.random.default_rng(0).random((64, 64, 3)).astype("float32")
        )
        out_a = aug(image, training=False).numpy()
        out_b = aug(image, training=False).numpy()
        np.testing.assert_array_equal(out_a, out_b)
