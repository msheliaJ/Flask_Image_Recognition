# Hand Digit Image Recognition

A Flask web app that predicts the digit (0–9) shown in a hand-sign image, using a
Keras model trained with transfer learning (`digit_model.h5`).

This fork adds code-quality checks (Pylint, pre-commit), a pytest suite and a
GitHub Actions CI workflow to the original project by Sachin-crypto.

## Requirements

- **Python 3.10** (TensorFlow 2.10 does not support Python 3.11 or newer)
- Git

## Setup

```bash
git clone <your-fork-url>
cd Flask_Image_Recognition

python3.10 -m venv venv          # Windows: py -3.10 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

pip install -r requirements.txt          # app only
pip install -r requirements-dev.txt      # app + Pylint, pytest, pre-commit
```

## Run

```bash
python app.py                    # http://127.0.0.1:9000
# or
flask --app app.py run --debug   # http://127.0.0.1:5000
```

Open the page, choose an image from `test_images/<digit>/`, and click **Submit**.

## Code quality

```bash
pylint app.py model.py tests     # configured in .pylintrc (fail-under 9.0)
pre-commit install               # run checks automatically on every commit
pre-commit run --all-files       # run every hook manually
```

## Tests

```bash
pytest                           # all tests
pytest -m "not slow"             # skip the tests that run the real model
```

Route tests mock the model so they run quickly; tests marked `slow` load
`digit_model.h5` and check predictions on sample images.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:
checkout → Python 3.10 → install dependencies → Pylint → pytest.
