from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

IMG_SIZE = 128

# 🔥 Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = gray / 255.0
    gray = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32')

    return gray


@app.route("/")
def home():
    return "✅ TFLite API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        img = preprocess_image(file)

        # 🔥 TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        result = "Real Banknote ✅" if prediction > 0.6 else "Fake Banknote ❌"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)