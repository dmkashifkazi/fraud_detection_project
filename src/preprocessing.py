
import sqlite3
import pandas as pd



def load_data(db_path, table_name="fraud_detection", limit=None):
    """Load fraud detection data from a SQLite database."""

    connection = sqlite3.connect(db_path)

    try:
        query = f"SELECT * FROM {table_name}"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        data = pd.read_sql_query(query, connection)

    finally:
        connection.close()

    return data
"""
limit=None means load all rows when no limit is provided.
limit=5 means load only five rows for testing.
"""



def select_columns(data):
    """Keep the required features and target for fraud detection."""

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ]

    data = data[required_columns].copy()

    # Treat missing transaction types as a separate category
    data["type"] = data["type"].fillna("MISSING")

    return data


def split_data(data):
    """Split data into training, evaluation, and live partitions."""

    train_end = 4_000_000
    eval_end = 5_000_000

    if len(data) <= train_end:
        raise ValueError(
            f"Dataset has {len(data):,} rows; "
            f"at least {train_end + 1:,} rows are needed."
        )

    train_raw = data.iloc[:train_end].copy()
    eval_raw = data.iloc[train_end:eval_end].copy()
    live_raw = data.iloc[eval_end:].copy()

    return train_raw, eval_raw, live_raw



