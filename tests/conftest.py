"""Shared pytest fixtures for the Flask app and generated test images."""
# pylint: disable=redefined-outer-name
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app import app as flask_app

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "test_images"


def make_image_bytes(mode="RGB", size=(64, 64), fmt="PNG"):
    """Build an in-memory image file so tests do not depend on files on disk."""
    colour = 128 if mode in ("L", "1") else (120,) * len(mode)
    buffer = BytesIO()
    Image.new(mode, size, colour).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


@pytest.fixture
def client():
    """Flask test client with TESTING enabled."""
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def mock_predict(monkeypatch):
    """Replace the model call in app.py with a fake that always returns 7.

    Records the arrays it receives so tests can check what the route
    passed to the model, without loading or running TensorFlow.
    """
    calls = []

    def fake_predict(img_array):
        calls.append(img_array)
        return 7

    monkeypatch.setattr("app.predict_result", fake_predict)
    return calls


def sample_image(digit):
    """Path of the first real sample image for a digit (0-9)."""
    return sorted((SAMPLE_DIR / str(digit)).glob("*.jpeg"))[0]
