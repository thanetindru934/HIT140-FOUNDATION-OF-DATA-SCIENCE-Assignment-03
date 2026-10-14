
import pandas as pd
import math
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# READ DATASET


BASE = Path(__file__).resolve().parent

FILE = BASE / "dataset_2_1_final.csv"

df = pd.read_csv(FILE, encoding="utf-8-sig")

print("Dataset loaded successfully!")
print("Total matches:", len(df))
print("Columns:", df.columns.tolist())


"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
OBJECTIVE 2.1: PREDICTING GOAL DIFFERENCE
"""

# SELECT X1 TO X8 AND Y


# Use exactly 8 explanatory variables
# Columns H to O contain X1-X8
model_df = df.iloc[:, 7:15].copy()

# Put actual goal difference (Y) as the last column
model_df["goal_difference"] = df.iloc[:, 6]

# Check for missing values
if model_df.isnull().any().any():
    raise ValueError(
        "Missing values found in X1-X8 or Y."
    )

# Here we have used  iloc method
x = model_df.iloc[:,:-1].values
y = model_df.iloc[:,-1].values

# Get explanatory variable names
x_columns = model_df.columns[:-1]


# SPLIT DATASET


# 60% training and 40% testing
# Keep row indices to identify test matches
row_indices = np.arange(len(df))

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    x,
    y,
    row_indices,
    test_size=0.4,
    random_state=0
)

print("\nTraining matches:", len(X_train))
print("Testing matches:", len(X_test))


# BUILD LINEAR REGRESSION MODEL


model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Print intercept and coefficients
print("\nIntercept:", model.intercept_)

print("\nRegression Coefficients:")
for variable, coefficient in zip(x_columns, model.coef_):
    print(f"{variable}: {coefficient:.6f}")


# PREDICT Y-HAT


# Predict goal difference 
y_pred = model.predict(X_test)

# Create actual vs predicted table
df_pred = pd.DataFrame({
    "Actual": y_test,
    "Predicted": np.round(y_pred, 2)
})

print("\nActual vs Predicted Goal Difference:")
print(df_pred.to_string(index=False))

# Save test predictions to CSV
PREDICTIONS_FILE = BASE / "FIFA_World_Cup_2026_Objective_2.1_predictions.csv"

df_pred.to_csv(
    PREDICTIONS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# 6. LINEAR REGRESSION PERFORMANCE


mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = math.sqrt(mse)

# Normalised RMSE
y_max = y.max()
y_min = y.min()

if y_max == y_min:
    raise ValueError("Cannot calculate Normalised RMSE: Y range is zero.")

rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)

print("\nLinear Regression Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("RMSE (Normalised):", rmse_norm)
print("R^2:", r_2)


"""
COMPARE LINEAR REGRESSION VS BASELINE MODEL
BASELINE USES THE MEAN OF TRAINING Y
"""


# BASELINE MODEL


# Mean of training goal differences
y_base = np.mean(y_train)

# Predict same mean for every test match
y_pred_base = [y_base] * len(y_test)

# Show baseline actual vs predicted
df_base_pred = pd.DataFrame({
    "Actual Y": y_test,
    "Baseline Predicted": y_pred_base
})

print("\nBaseline Actual vs Predicted:")
print(df_base_pred.to_string(index=False))


#  BASELINE PERFORMANCE


mae_base = metrics.mean_absolute_error(
    y_test, y_pred_base
)

mse_base = metrics.mean_squared_error(
    y_test, y_pred_base
)

rmse_base = math.sqrt(mse_base)

rmse_norm_base = rmse_base / (y_max - y_min)

r_2_base = metrics.r2_score(
    y_test, y_pred_base
)

print("\nBaseline Performance:")
print("MAE:", mae_base)
print("MSE:", mse_base)
print("RMSE:", rmse_base)
print("RMSE (Normalised):", rmse_norm_base)
print("R^2:", r_2_base)


# COMPARE BOTH MODELS


comparison = pd.DataFrame({
    "metric": [
        "MAE",
        "MSE",
        "RMSE",
        "RMSE (Normalised)",
        "R^2"
    ],
    "linear_regression": [
        mae, mse, rmse, rmse_norm, r_2
    ],
    "baseline": [
        mae_base,
        mse_base,
        rmse_base,
        rmse_norm_base,
        r_2_base
    ]
})

# Round results to 3 decimal places
comparison = comparison.round(3)

print("\nModel Performance Comparison:")
print(comparison.to_string(index=False))

# Save comparison results
COMPARISON_FILE = BASE / "FIFA_World_Cup_2026_Objective_2.1_model_comparison.csv"

comparison.to_csv(
    COMPARISON_FILE,
    index=False,
    encoding="utf-8-sig"
)


# FINAL OUTPUT


print("\nModel evaluation completed successfully!")
print("Test predictions saved to:", PREDICTIONS_FILE)
print("Model comparison saved to:", COMPARISON_FILE)
