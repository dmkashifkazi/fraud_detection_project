from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,average_precision_score)


def train_decision_tree(X_train, y_train):
    """Train a Decision Tree model for comparison."""

    dt_model = DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )

    dt_model.fit(X_train, y_train)

    return dt_model


def train_random_forest(X_train, y_train):
    """Train the main Random Forest fraud detection model."""

    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=2,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    return rf_model


def evaluate_model(model, X, y, model_name):
    """Evaluate a trained model using classification metrics.
    The parameter zero_division=0 tells scikit-learn to return 0 
    instead of raising a warning or producing an undefined metric 
    when the calculation has a zero denominator.
    """

    y_pred = model.predict(X)

    print(f"===== {model_name} — Evaluation Set =====")
    print(f"Accuracy : {accuracy_score(y, y_pred):.6f}")
    print(f"Precision: {precision_score(y, y_pred, zero_division=0):.6f}")
    print(f"Recall   : {recall_score(y, y_pred, zero_division=0):.6f}")
    print(f"F1 Score : {f1_score(y, y_pred, zero_division=0):.6f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    return y_pred



