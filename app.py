
from flask import Flask, request, jsonify
import pandas as pd

from src.model_prediction import predict_transactions

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Financial Fraud Detection API is running"
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        transaction = request.get_json()

        if not transaction:
            return jsonify({"error": "Transaction data is required"}), 400

        data = pd.DataFrame([transaction])
        results = predict_transactions(data)
        prediction = int(results["predictedFraud"].iloc[0])

        return jsonify({
            "prediction": prediction,
            "result": "Fraud" if prediction == 1 else "Not Fraud"
        })

    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True)
