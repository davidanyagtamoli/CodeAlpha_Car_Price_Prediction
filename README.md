# Car Price Prediction

This project is part of my CodeAlpha Data Science Internship.

The aim of the project is to build a machine learning model that can predict the selling price of a used car based on information such as its year, current price, mileage, fuel type, selling type, transmission, and previous owners.

## Project Overview

I started by cleaning and exploring the dataset before training different regression models.

The main steps I followed were:

* Cleaned duplicate records and unnecessary whitespace
* Created a `Car_Age` feature during the analysis
* Explored the relationships between the features and selling price
* Selected the features to use for the final model
* Prepared the numerical and categorical features
* Tested different regression models
* Compared the models using MAE, RMSE, and R²
* Selected the best-performing model
* Saved the trained model using Joblib
* Built a simple Streamlit app for making predictions

## Dataset

The original dataset contained **301 rows**. After removing 2 duplicate records, the cleaned dataset contains **299 rows**.

The target variable is:

* `Selling_Price`

The features used by the final model are:

* `Year`
* `Present_Price`
* `Driven_kms`
* `Fuel_Type`
* `Selling_type`
* `Transmission`
* `Owner`

I also created `Car_Age` during the analysis, but I did not use it in the final model because it is directly derived from `Year`.

`Car_Name` was also left out of the final model because the dataset is relatively small and contains many different car names.

## Exploratory Data Analysis

One of the clearest findings from the analysis was the relationship between `Present_Price` and `Selling_Price`.

The correlation between them was about **0.876**, which was much stronger than the correlation between `Car_Age` and `Selling_Price`.

I also found that older cars generally tend to have lower selling prices, although there are some exceptions.

## Models Tested

I compared several regression models using the same train/test split.

| Model                                  |    MAE |   RMSE |     R² |
| -------------------------------------- | -----: | -----: | -----: |
| Present Price Only - Linear Regression | 1.9233 | 3.2228 | 0.5970 |
| Linear Regression                      | 1.4725 | 2.5245 | 0.7527 |
| Random Forest                          | 1.5837 | 3.8812 | 0.4155 |
| Gradient Boosting                      | 1.4999 | 3.2020 | 0.6022 |
| Ridge Regression (alpha=10)            | 1.4715 | 2.4778 | 0.7618 |

Ridge Regression with `alpha=10` gave the best results among the models I tested.

## Final Model

The final model is **Ridge Regression** with `alpha=10`.

The test-set results were:

* **MAE:** 1.4715
* **RMSE:** 2.4778
* **R²:** 0.7618

The model explains about **76% of the variation** in the test-set selling prices.

The preprocessing and model are kept together in a scikit-learn pipeline. This means the same preprocessing steps are automatically applied when making predictions on new cars.

## Making Predictions

The project includes a Python prediction script that loads the saved model and predicts the selling price of an example car.

Run:

```bash
python src/predict.py
```

The model is saved as:

```text
models/car_price_ridge.pkl
```

## Streamlit App

I also built a simple Streamlit web app so that predictions can be made through an interactive interface instead of entering values directly in Python.

Start the app with:

```bash
streamlit run app/app.py
```

The app allows you to enter:

* Year
* Present Price
* Driven Kilometres
* Fuel Type
* Selling Type
* Transmission
* Previous Owners

It then returns an estimated selling price.

**Live Demo:** https://codealphacarpriceprediction-1.streamlit.app/

## Limitations

The dataset is fairly small, so the model may not perform equally well for every used car.

Some unusual cars were difficult for the model to predict. For example, the model underestimated one high-priced car by a large amount and overestimated some cheaper cars.

The Ridge model can also produce a negative prediction for some extreme inputs, even though a car cannot actually have a negative selling price. This is a limitation of the current model.

## Project Structure

```text
CodeAlpha_Car_Price_Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── car data.csv
│   └── processed/
│       └── car_data_cleaned.csv
│
├── models/
│   └── car_price_ridge.pkl
│
├── notebooks/
│   └── 01_data_inspection_and_eda.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

To train the model again:

```bash
python src/train.py
```

The trained model will be saved to:

```text
models/car_price_ridge.pkl
```

## Technologies

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Git & GitHub

## Internship

This project was completed as part of my **CodeAlpha Data Science Internship**.
