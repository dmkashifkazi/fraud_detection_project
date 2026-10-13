
from src.preprocessing import load_data, select_columns
from src.feature_engineering import handle_missing_values, encode_features

db_path = "data/Classification.db"

# Step 1: Load a small sample
data = load_data(db_path, limit=1000)
data = select_columns(data)

print("Sample data shape:", data.shape)

# Step 2: Split the sample for testing only
train_raw = data.iloc[:600].copy()
eval_raw = data.iloc[600:800].copy()
live_raw = data.iloc[800:].copy()

print("Training rows:", len(train_raw))
print("Evaluation rows:", len(eval_raw))
print("Live rows:", len(live_raw))

# Step 3: Handle missing numerical values
train_raw, eval_raw, live_raw, medians = handle_missing_values(
    train_raw, eval_raw, live_raw
)

print("Medians:", medians)

# Step 4: Encode categorical features
target = "isFraud"

X_train, y_train, X_eval, y_eval, X_live, y_live = encode_features(
    train_raw, eval_raw, live_raw, target
)

# Step 5: Verify the output
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_eval shape:", X_eval.shape)
print("y_eval shape:", y_eval.shape)
print("X_live shape:", X_live.shape)
print("y_live shape:", y_live.shape)

print("Training and evaluation columns match:",
      list(X_train.columns) == list(X_eval.columns))

print("Training and live columns match:",
      list(X_train.columns) == list(X_live.columns))

print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_eval:", X_eval.isnull().sum().sum())
print("Missing values in X_live:", X_live.isnull().sum().sum())