from flask import Flask, request, jsonify
from sklearn.datasets import load_iris   
import numpy as np
import joblib

iris = load_iris() 
app = Flask(__name__)
model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("std_scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    flower = np.array(data["flower"]).reshape(1,4)
    flower = scaler.transform(flower)
    prediction = model.predict(flower)
    predicted_class = str(iris.target_names[prediction])
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)