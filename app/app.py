import os
import joblib
import pandas as pd
import streamlit as st


# Find the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "car_price_ridge.pkl"
)

model = joblib.load(MODEL_PATH)


# Page setup
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# Title and introduction
st.title("🚗 Car Price Predictor")
st.write(
    "Enter the details of a used car to estimate its selling price."
)

st.info(
    "This prediction is based on a Ridge Regression model trained on "
    "used-car data."
)


# Input section
st.subheader("Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2026,
        value=2015,
        step=1
    )

    present_price = st.number_input(
        "Present Price (lakh)",
        min_value=0.0,
        value=5.5,
        step=0.1
    )

    driven_kms = st.number_input(
        "Driven Kilometres",
        min_value=0,
        value=30000,
        step=1000
    )

    owner = st.number_input(
        "Previous Owners",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )

with col2:
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    selling_type = st.selectbox(
        "Selling Type",
        ["Dealer", "Individual"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )


# Prediction button
st.divider()

if st.button("Predict Selling Price", use_container_width=True):

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

    st.subheader("Prediction")

    if prediction < 0:
        st.warning(
            "The model produced a negative estimate for this input. "
            "This can happen with unusual combinations of features."
        )
    else:
        st.success(
            f"Estimated Selling Price: {prediction:.2f} lakh"
        )


# Model information
st.divider()

st.subheader("About the Model")

st.write(
    "The final model is Ridge Regression with alpha=10. "
    "It uses the car's year, present price, driven kilometres, "
    "fuel type, selling type, transmission, and number of previous owners."
)

st.caption(
    "Test-set performance: MAE 1.4715 | RMSE 2.4778 | R² 0.7618"
)
