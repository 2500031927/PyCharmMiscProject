import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score

)

from ML.Program.third import comparison

print("Libraries imported successsfully.")
df = pd.read_csv("placement_predict_50k Dataset.csv")
print("Dataset loaded successfully!")
print("Rows   :", df.shape[0])
print("Columns   :", df.shape[1])
print("First 5 rows:")
print(df.head())


print("\nColumn names:")
print(df.columns.tolist())

print("CGPA column exists: ", "CGPA" in df.columns)
print("Salary Package column exists: ", "Salary Package" in df.columns)

if "CGPA" not in df.columns or "Salary Package" not in df.columns:
    raise ValueError("CGPA or Salary Package column is missing!")
X = df[["CGPA"]]
Y = df["Salary Package"]

print("Input feature:", X.columns.tolist())
print("Target:", Y.name)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
)

print("Training samples", len(X_train))
print("Testing samples", len(X_test))
model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ]
)

print("Linear regression model created.")
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Salary": Y_test.values,
    "Predicted Salary": y_pred
})

print(comparison)
mae = mean_absolute_error(Y_test, y_pred)
print("MAE:", round(mae, 4))
print()
mse = mean_squared_error(Y_test, y_pred)
print("MSE:", round(mse, 4))
print()
rmse = mean_squared_error(Y_test, y_pred) ** 0.5
print("RMSE:", round(rmse, 4))
print()
r2 = r2_score(Y_test, y_pred)
print("R2:", round(r2, 4))
print()






