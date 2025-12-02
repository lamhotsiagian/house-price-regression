# house_price_regression.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_dataset(seed: int = 42, size: int = 100) -> pd.DataFrame:
    """
    Create a synthetic housing dataset using NumPy
    and return it as a Pandas DataFrame.
    """
    np.random.seed(seed)

    # Features
    sqft = np.random.randint(800, 3500, size)        # square footage
    bedrooms = np.random.randint(1, 5, size)         # number of bedrooms
    age = np.random.randint(1, 30, size)             # house age in years

    # Target (price) = linear combination + noise
    price = (
        sqft * 250
        + bedrooms * 20000
        - age * 1500
        + np.random.randint(-20000, 20000, size)     # random noise
    )

    df = pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "age": age,
        "price": price
    })

    return df


def train_model(df: pd.DataFrame):
    """
    Split the data, train a linear regression model,
    and return the trained model plus train/test sets.
    """
    # Features and target
    X = df[["sqft", "bedrooms", "age"]]
    y = df["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using RMSE and print coefficients.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("=== Model Evaluation ===")
    print(f"RMSE: {rmse:,.2f}")
    print("Coefficients (sqft, bedrooms, age):", model.coef_)
    print("Intercept:", model.intercept_)

    return rmse, y_pred


def predict_new_house(model):
    """
    Use the trained model to predict the price of a new house.
    """
    new_house = pd.DataFrame({
        "sqft": [2000],
        "bedrooms": [3],
        "age": [10]
    })

    pred_price = model.predict(new_house)[0]

    print("\n=== Prediction for New House ===")
    print("Input features:")
    print(new_house)
    print(f"Predicted price: ${pred_price:,.2f}")

    return pred_price


def main():
    # 1. Create dataset
    df = create_dataset()
    print("=== Head of Dataset ===")
    print(df.head(), "\n")

    # 2. Train model
    model, X_train, X_test, y_train, y_test = train_model(df)

    # 3. Evaluate model
    evaluate_model(model, X_test, y_test)

    # 4. Predict on new example
    predict_new_house(model)


if __name__ == "__main__":
    main()
