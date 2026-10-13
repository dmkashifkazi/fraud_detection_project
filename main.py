import joblib
from sklearn.metrics import average_precision_score
from src.preprocessing import load_data, select_columns, split_data
from src.feature_engineering import handle_missing_values, encode_features
from src.model_train import (
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    # Step 1: Load the dataset
    db_path = "data/Classification.db"
    data = load_data(db_path)

    print("Dataset shape:", data.shape)

    # Step 2: Select the required columns
    data = select_columns(data)

    print("Working dataset shape:", data.shape)

    
    # Step 3: Split into training, evaluation, and live datasets
    train_raw, eval_raw, live_raw = split_data(data)

    print("Training rows:", train_raw.shape)
    print("Evaluation rows:", eval_raw.shape)
    print("Live rows:", live_raw.shape)

    
    # Step 4: Handle missing numerical values
    train_raw, eval_raw, live_raw, medians = handle_missing_values(
        train_raw, eval_raw, live_raw
    )

    print("Training medians:", medians)

    
    # Step 5: One-hot encode the transaction type
    target = "isFraud"

    X_train, y_train, X_eval, y_eval, X_live, y_live = encode_features(
        train_raw, eval_raw, live_raw, target
    )

    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_eval:", X_eval.shape, "| y_eval:", y_eval.shape)
    print("X_live:", X_live.shape, "| y_live:", y_live.shape)


   
    # Step 7: Train the Decision Tree model

    dt_model = train_decision_tree(X_train, y_train)

    print("Decision Tree model training completed.")


    # Step 8: Evaluate the Decision Tree model

    evaluate_model(dt_model, X_eval, y_eval, "Decision Tree")


    # Step 9: Train the Random Forest model

    rf_model = train_random_forest(X_train, y_train)

    print("Random Forest model training completed.")


    # Step 10: Evaluate the Random Forest model

    evaluate_model(rf_model, X_eval, y_eval, "Random Forest")

    # Step 11: Calculate PR-AUC for Random Forest

    y_eval_proba = rf_model.predict_proba(X_eval)[:, 1]
    pr_auc = average_precision_score(y_eval, y_eval_proba)

    print(f"Random Forest PR-AUC: {pr_auc:.6f}")

    # Step 12: Save the trained Random Forest model

    joblib.dump(rf_model, "models/random_forest_model.pkl")

    print("Random Forest model saved successfully.")


    # Step 13: Save preprocessing information

    metadata_path = "models/fraud_detection_preprocessing.pkl"

    preprocessing_info = {
        "median_newbalanceOrig": medians["newbalanceOrig"],
        "median_oldbalanceDest": medians["oldbalanceDest"],
        "feature_columns": list(X_train.columns),
        "categorical_column": "type",
        "missing_type_label": "MISSING",
        "target_column": target
    }

    joblib.dump(preprocessing_info, metadata_path)

    print("Preprocessing information saved successfully.")


    
if __name__ == "__main__":
    main()







