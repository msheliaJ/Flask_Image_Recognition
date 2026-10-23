"""Flask web app that predicts hand-sign digits from uploaded images."""
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)

NO_FILE_MSG = "No file was uploaded."
BAD_FILE_MSG = "File cannot be processed."


@app.route("/")
def main():
    """Render the upload page."""
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Preprocess the uploaded image, run the model and render the result.

    Returns 400 with an error message when no file is sent or the file is
    not a readable image, so clients can tell failures from predictions.
    """
    upload = request.files.get('file')
    if upload is None or upload.filename == "":
        return render_template("result.html", err=NO_FILE_MSG), 400

    try:
        img = preprocess_img(upload.stream)
    # OSError covers PIL.UnidentifiedImageError and truncated images.
    except (OSError, ValueError):
        return render_template("result.html", err=BAD_FILE_MSG), 400

    pred = predict_result(img)
    return render_template("result.html", predictions=str(pred))


if __name__ == "__main__":
    app.run(port=9000, debug=True)
