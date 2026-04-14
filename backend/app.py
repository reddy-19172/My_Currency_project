from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load trained model
model = load_model("model.h5")

IMG_SIZE = 128

def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    edges = edges / 255.0
    edges = edges.reshape(IMG_SIZE, IMG_SIZE, 1)

    return edges

@app.route("/")
def home():
    return "✅ Banknote API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        img = preprocess_image(file)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        result = "Real Banknote ✅" if prediction > 0.5 else "Fake Banknote ❌"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)