from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("img_classifier.keras")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image = np.array(data["image"]).reshape(1, 28, 28, 1) / 255.0  # Normalize if needed
    prediction = model.predict(image)
    predicted_class = int(np.argmax(prediction))
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
