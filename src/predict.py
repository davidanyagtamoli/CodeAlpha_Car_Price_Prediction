import os
import joblib
import pandas as pd


# Find the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "car_price_ridge.pkl"
)

model = joblib.load(MODEL_PATH)


def predict_price(
    year,
    present_price,
    driven_kms,
    fuel_type,
    selling_type,
    transmission,
    owner
):
    """Predict the selling price of a used car."""

    car_data = pd.DataFrame([{
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    prediction = model.predict(car_data)[0]

    return prediction


if __name__ == "__main__":

    predicted_price = predict_price(
        year=2015,
        present_price=5.5,
        driven_kms=30000,
        fuel_type="Petrol",
        selling_type="Dealer",
        transmission="Manual",
        owner=0
    )

    print(f"Predicted Selling Price: {predicted_price:.2f}")
