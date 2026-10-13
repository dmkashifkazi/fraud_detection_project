import joblib
import pandas as pd


def load_model_and_metadata():
    """Load the trained model and preprocessing information."""

    model = joblib.load("models/random_forest_model.pkl")

    preprocessing_info = joblib.load(
        "models/fraud_detection_preprocessing.pkl"
    )

    return model, preprocessing_info


def prepare_prediction_data(data, preprocessing_info):
    """Prepare transaction data using the saved preprocessing information."""

    data = data.copy()

    # Keep the required columns
    features = [
        "step", "type", "amount", "oldbalanceOrg",
        "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    ]
    data = data[features]

    # Handle missing values
    data["type"] = data["type"].fillna(
        preprocessing_info["missing_type_label"]
    )

    data["newbalanceOrig"] = data["newbalanceOrig"].fillna(
        preprocessing_info["median_newbalanceOrig"]
    )
    data["oldbalanceDest"] = data["oldbalanceDest"].fillna(
        preprocessing_info["median_oldbalanceDest"]
    )

    # One-hot encode transaction type
    data = pd.get_dummies(data, columns=["type"], dtype=int)

    # Match the feature columns used during training
    data = data.reindex(
        columns=preprocessing_info["feature_columns"],
        fill_value=0
    )

    return data


def predict_transactions(data):
    """Predict whether transactions are fraudulent."""

    model, preprocessing_info = load_model_and_metadata()

    X = prepare_prediction_data(data, preprocessing_info)

    predictions = model.predict(X)

    results = data.copy()
    results["predictedFraud"] = predictions

    return results

