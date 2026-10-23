"""Loads the trained Keras model and provides preprocessing and prediction helpers."""
from pathlib import Path

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "digit_model.h5"
IMG_SIZE = (224, 224)

# Loaded once at import time and reused for every request.
model = load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Open an image and convert it to a normalised (1, 224, 224, 3) array.

    Images are converted to RGB first so greyscale, palette and transparent
    (RGBA) uploads match the three-channel input the model was trained on.
    """
    with Image.open(img_path) as op_img:
        img_resize = op_img.convert("RGB").resize(IMG_SIZE)
    img2arr = img_to_array(img_resize) / 255.0
    return img2arr.reshape(1, *IMG_SIZE, 3)


def predict_result(predict):
    """Return the predicted digit (index of the highest class score)."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
