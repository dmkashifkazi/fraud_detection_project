import pandas as pd


def handle_missing_values(train_raw, eval_raw, live_raw):
    """Fill missing numerical values using training medians."""

    numeric_to_impute = ["newbalanceOrig", "oldbalanceDest"]

    medians = {}

    for col in numeric_to_impute:
        medians[col] = train_raw[col].median()

    for part in (train_raw, eval_raw, live_raw):
        for col, median_value in medians.items():
            part[col] = part[col].fillna(median_value)

    return train_raw, eval_raw, live_raw, medians





def encode_features(train_raw, eval_raw, live_raw, target):
    """One-hot encode transaction types consistently across all partitions.
       Separate features (X) and target (y)
    """

    X_train_raw = train_raw.drop(columns=[target])
    y_train = train_raw[target].astype(int)

    X_eval_raw = eval_raw.drop(columns=[target])
    y_eval = eval_raw[target].astype(int)

    X_live_raw = live_raw.drop(columns=[target])
    y_live = live_raw[target].astype(int)

    # Use the same categories and feature columns for all partitions
    all_X = pd.concat([X_train_raw, X_eval_raw, X_live_raw], axis=0)
    all_X = pd.get_dummies(all_X, columns=["type"], dtype=int)

    X_train = all_X.iloc[:len(X_train_raw)].copy()

    X_eval = all_X.iloc[
        len(X_train_raw):len(X_train_raw) + len(X_eval_raw)
    ].copy()

    X_live = all_X.iloc[
        len(X_train_raw) + len(X_eval_raw):
    ].copy()

    return X_train, y_train, X_eval, y_eval, X_live, y_live
