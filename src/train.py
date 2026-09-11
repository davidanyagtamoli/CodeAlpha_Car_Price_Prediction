import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Find the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load cleaned dataset
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "car_data_cleaned.csv"
)

df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Final features selected during modeling
numeric_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Final Ridge model
final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=10.0))
    ]
)

# Same train/test split used during evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
final_model.fit(X_train, y_train)

# Save model
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "car_price_ridge.pkl"
)

joblib.dump(final_model, MODEL_PATH)

print("Model trained successfully.")
print("Model: Ridge Regression")
print("Alpha: 10.0")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Model saved to: {MODEL_PATH}")
