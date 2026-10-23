"""Unit tests for preprocessing and prediction in model.py."""
import numpy as np
import pytest
from PIL import UnidentifiedImageError

import model
from tests.conftest import make_image_bytes, sample_image


# ---------- preprocess_img ----------

def test_preprocess_real_sample_shape_and_range():
    """A real sample image becomes a (1, 224, 224, 3) array scaled to [0, 1]."""
    arr = model.preprocess_img(sample_image(2))
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


@pytest.mark.parametrize("size", [(1, 1), (100, 400), (1024, 768)])
def test_preprocess_resizes_any_size(size):
    """Tiny, tall and large images are all resized to 224x224."""
    arr = model.preprocess_img(make_image_bytes(size=size))
    assert arr.shape == (1, 224, 224, 3)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_preprocess_converts_to_three_channels(mode):
    """Alpha, greyscale and palette images are converted to RGB."""
    arr = model.preprocess_img(make_image_bytes(mode=mode))
    assert arr.shape == (1, 224, 224, 3)


def test_preprocess_scales_pixel_values():
    """A uniform pixel value of 120 becomes 120/255 after normalisation."""
    arr = model.preprocess_img(make_image_bytes())
    assert np.allclose(arr, 120 / 255.0)


def test_preprocess_missing_path_raises():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        model.preprocess_img("no/such/image.jpeg")


def test_preprocess_non_image_raises():
    """A text file is rejected by PIL."""
    with pytest.raises(UnidentifiedImageError):
        model.preprocess_img("invalid_file.txt")


# ---------- predict_result (model mocked) ----------

class FakeModel:  # pylint: disable=too-few-public-methods
    """Stand-in for the Keras model that returns fixed class scores."""

    def __init__(self, scores):
        self.scores = np.array([scores])
        self.received = None

    def predict(self, batch, **_kwargs):
        """Record the input and return the fixed scores."""
        self.received = batch
        return self.scores


def test_predict_result_returns_argmax(monkeypatch):
    """predict_result returns the index of the highest score."""
    scores = [0.01] * 10
    scores[4] = 0.9
    fake = FakeModel(scores)
    monkeypatch.setattr(model, "model", fake)

    batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    assert model.predict_result(batch) == 4
    assert fake.received is batch


def test_predict_result_tie_picks_first(monkeypatch):
    """With equal scores, argmax picks the lowest digit (edge case)."""
    monkeypatch.setattr(model, "model", FakeModel([0.1] * 10))
    assert model.predict_result(np.zeros((1, 224, 224, 3))) == 0


# ---------- Real model (slow) ----------

@pytest.mark.slow
@pytest.mark.parametrize("digit", [0, 3, 5, 9])
def test_real_model_predicts_sample_digits(digit):
    """End-to-end check with the trained model on known sample images."""
    prediction = model.predict_result(model.preprocess_img(sample_image(digit)))
    assert isinstance(prediction, (int, np.integer))
    assert int(prediction) == digit
