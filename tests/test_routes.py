"""Tests for the Flask routes: home page and /prediction."""
from io import BytesIO

import pytest

from tests.conftest import make_image_bytes

ERROR_MSG = b"File cannot be processed."


def post_file(client, file_obj, filename):
    """POST a file to /prediction as a multipart form upload."""
    return client.post(
        "/prediction",
        data={"file": (file_obj, filename)},
        content_type="multipart/form-data",
    )


# ---------- Home page ----------

def test_home_page_loads(client):
    """GET / returns 200 and the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="file"' in response.data
    assert b'action="/prediction"' in response.data


def test_prediction_rejects_get(client):
    """/prediction only accepts POST."""
    assert client.get("/prediction").status_code == 405


def test_unknown_route_returns_404(client):
    """Unknown URLs are not served."""
    assert client.get("/does-not-exist").status_code == 404


# ---------- Valid requests (model mocked) ----------

def test_valid_image_returns_prediction(client, mock_predict):
    """A valid JPEG produces a result page showing the model's digit."""
    response = post_file(client, make_image_bytes(fmt="JPEG"), "hand.jpg")
    assert response.status_code == 200
    assert b">7</h2>" in response.data
    assert ERROR_MSG not in response.data
    assert len(mock_predict) == 1


def test_route_passes_model_shaped_array(client, mock_predict):
    """The route hands the model a (1, 224, 224, 3) normalised array."""
    post_file(client, make_image_bytes(size=(500, 300)), "big.png")
    arr = mock_predict[0]
    assert arr.shape == (1, 224, 224, 3)
    assert 0.0 <= arr.min() and arr.max() <= 1.0


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_png_with_alpha_or_greyscale_is_accepted(client, mock_predict, mode):
    """Transparent PNGs and greyscale images are valid uploads too."""
    response = post_file(client, make_image_bytes(mode=mode), f"{mode}.png")
    assert response.status_code == 200
    assert b">7</h2>" in response.data
    assert len(mock_predict) == 1


# ---------- Missing / invalid input ----------

def test_missing_file_field(client, mock_predict):
    """No 'file' field in the form -> 400 with a clear message, model not called."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"No file was uploaded." in response.data
    assert not mock_predict


def test_empty_filename(client, mock_predict):
    """Submitting the form without choosing a file -> 400, model not called."""
    response = post_file(client, BytesIO(b""), "")
    assert response.status_code == 400
    assert b"No file was uploaded." in response.data
    assert not mock_predict


@pytest.mark.parametrize(
    "payload, filename",
    [
        (b"this is not an image", "notes.txt"),
        (b"fake_image_data", "fake.jpg"),        # right extension, wrong content
        (b"\xff\xd8\xff\xe0" + b"\x00" * 20, "truncated.jpg"),  # JPEG header only
    ],
)
def test_invalid_image_content(client, mock_predict, payload, filename):
    """Non-image or corrupt uploads -> 400 error page, model not called."""
    response = post_file(client, BytesIO(payload), filename)
    assert response.status_code == 400
    assert ERROR_MSG in response.data
    assert not mock_predict
