# House Price Regression (Pandas + NumPy + Scikit-Learn)

This project demonstrates a simple machine-learning workflow to predict house prices using **Linear Regression**.
The dataset is generated synthetically using **NumPy**, processed with **Pandas**, and modeled with **Scikit-Learn**.
Configuration values (dataset size and random seed) are stored in a `.env` file for flexibility.

---

## How to Run

```bash
pip install -r requirements.txt
python house_price_regression.py
```

---

## Configuration via `.env`

```
DATASET_SIZE=200
RANDOM_SEED=123
```

Update these values at any time — no need to edit the Python script.

---

## What the script does

* Creates a synthetic housing dataset
* Trains a Linear Regression model
* Evaluates model performance (RMSE + coefficients)
* Predicts the price of a new house example

---

## Files

| File                        | Purpose                         |
| --------------------------- | ------------------------------- |
| `house_price_regression.py` | Main ML pipeline                |
| `.env`                      | App configuration (size + seed) |
| `requirements.txt`          | Python dependencies             |

---

## 🧠 Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-Learn
* python-dotenv