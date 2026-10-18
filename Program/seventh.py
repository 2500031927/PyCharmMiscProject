import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# LOAD DATASET
# --------------------------------

file_path = "placement_predict_50k Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------
# DATA INFORMATION
# --------------------------------

print("\nDATA INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe().T)


# --------------------------------
# MISSING VALUE CHECK
# --------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTotal missing values:",
      df.isnull().sum().sum())


# --------------------------------
# DEFINE INPUT AND TARGET
# --------------------------------

y = df["Salary Package"]

X = df.drop(
    ["Salary Package", "PlacementStatus", "IsAnomaly"],
    axis=1
)


# --------------------------------
# REMOVE STUDENT ID
# --------------------------------

if "StudentID" in X.columns:
    X = X.drop("StudentID", axis=1)

print("\nInput columns:")
print(X.columns.tolist())

print("\nTarget column:")
print("Salary Package")


# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN TEST SPLIT")
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------
# IDENTIFY COLUMNS
# --------------------------------

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X_train.select_dtypes(
    include=["str", "object"]
).columns


# --------------------------------
# HANDLE MISSING VALUES
# --------------------------------

for column in numeric_columns:

    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(median_value)

    X_test[column] = X_test[column].fillna(median_value)


for column in categorical_columns:

    mode_value = X_train[column].mode()[0]

    X_train[column] = X_train[column].fillna(mode_value)

    X_test[column] = X_test[column].fillna(mode_value)


print("\nMissing values handled successfully.")


# --------------------------------
# ENCODING
# --------------------------------

for column in categorical_columns:

    encoder = LabelEncoder()

    X_train[column] = encoder.fit_transform(
        X_train[column].astype(str)
    )

    mapping = {
        value: index
        for index, value in enumerate(
            encoder.classes_
        )
    }

    X_test[column] = (
        X_test[column]
        .astype(str)
        .map(mapping)
        .fillna(-1)
    )


print("Encoding completed!")


# --------------------------------
# FEATURE SCALING
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")


# --------------------------------
# CREATE MODEL
# --------------------------------

model = LinearRegression()


# --------------------------------
# TRAIN MODEL
# --------------------------------

model.fit(
    X_train_scaled,
    y_train
)

print("\nModel training completed!")


# --------------------------------
# COEFFICIENTS
# --------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFEATURE COEFFICIENTS")

print(
    coefficients.sort_values(
        "Coefficient",
        ascending=False
    )
)


# --------------------------------
# INTERCEPT
# --------------------------------

print("\nIntercept:", model.intercept_)


# --------------------------------
# PREDICTION
# --------------------------------

y_pred = model.predict(X_test_scaled)

print("\nPredictions generated successfully.")


# --------------------------------
# ACTUAL VS PREDICTED
# --------------------------------

comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nACTUAL VS PREDICTED")

print(comparison.head(10))


# --------------------------------
# MODEL PERFORMANCE
# --------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMODEL PERFORMANCE")
print("-------------------------")

print("MAE :", round(mae, 3))
print("MSE :", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R²  :", round(r2, 3))


# --------------------------------
# ACTUAL VS PREDICTED GRAPH
# --------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2
)

plt.xlabel("Actual Salary Package")
plt.ylabel("Predicted Salary Package")

plt.title("Actual vs Predicted Salary")

plt.grid(True)

plt.show()


# ==========================================================
# GRADIENT DESCENT DEMONSTRATION
# CGPA -> Salary Package
# ==========================================================


# --------------------------------
# PREPARE SIMPLE DATA
# --------------------------------

X_simple = X_train[["CGPA"]].copy()

y_simple = y_train.values


# --------------------------------
# SCALE CGPA
# --------------------------------

simple_scaler = StandardScaler()

X_simple_scaled = simple_scaler.fit_transform(
    X_simple
).flatten()


# --------------------------------
# INITIAL PARAMETERS
# --------------------------------

w_initial = 0.0
b_initial = 0.0


# --------------------------------
# INITIAL COST
# --------------------------------

initial_predictions = (
    w_initial * X_simple_scaled
    + b_initial
)

initial_cost = np.mean(
    (initial_predictions - y_simple) ** 2
)


# --------------------------------
# GRADIENT DESCENT FUNCTION
# --------------------------------

def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=100
):

    w = 0.0
    b = 0.0

    n = len(y)

    cost_history = []
    weight_history = []
    bias_history = []

    for epoch in range(epochs):

        # 1. Make predictions
        y_pred = w * X + b

        # 2. Calculate error
        error = y_pred - y

        # 3. Calculate gradients
        dw = (2 / n) * np.sum(
            error * X
        )

        db = (2 / n) * np.sum(
            error
        )

        # 4. Update parameters
        w = w - learning_rate * dw

        b = b - learning_rate * db

        # 5. Calculate cost
        new_predictions = w * X + b

        cost = np.mean(
            (new_predictions - y) ** 2
        )

        # Store values
        cost_history.append(cost)
        weight_history.append(w)
        bias_history.append(b)

    return (
        w,
        b,
        cost_history,
        weight_history,
        bias_history
    )


# --------------------------------
# RUN GRADIENT DESCENT
# --------------------------------

w, b, cost_history, weight_history, bias_history = (
    gradient_descent(
        X_simple_scaled,
        y_simple,
        learning_rate=0.01,
        epochs=100
    )
)


# --------------------------------
# FINAL COST
# --------------------------------

final_cost = cost_history[-1]


# --------------------------------
# PRINT RESULTS
# --------------------------------

print("\nGRADIENT DESCENT RESULTS")
print("-------------------------")

print(
    "Initial Cost :",
    round(initial_cost, 3)
)

print(
    "Final Cost   :",
    round(final_cost, 3)
)

print()

print(
    "Initial Weight :",
    round(w_initial, 3)
)

print(
    "Final Weight   :",
    round(w, 3)
)

print()

print(
    "Initial Intercept :",
    round(b_initial, 3)
)

print(
    "Final Intercept   :",
    round(b, 3)
)


# --------------------------------
# GRADIENT DESCENT COST GRAPH
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 101),
    cost_history,
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")

plt.title(
    "Cost Reduction During Gradient Descent"
)

plt.grid(True)

plt.show()


# --------------------------------
# REGRESSION LINE AFTER GRADIENT DESCENT
# --------------------------------

x_line = np.linspace(
    X_simple_scaled.min(),
    X_simple_scaled.max(),
    100
)

y_final_line = w * x_line + b

plt.figure(figsize=(8, 5))

plt.scatter(
    X_simple_scaled,
    y_simple,
    alpha=0.2,
    label="Actual Data"
)

plt.plot(
    x_line,
    y_final_line,
    linewidth=3,
    label="Gradient Descent Line"
)

plt.xlabel("Standardized CGPA")

plt.ylabel("Salary Package")

plt.title(
    "Linear Regression Using Gradient Descent"
)
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost Reduction")
plt.grid(True)
plt.show()
plt.legend()

plt.grid(True)
plt.show()
y_final_line = w * x_line + b

plt.figure(figsize=(9, 6))

plt.scatter(
    X_simple_scaled,
    y_simple,
    alpha=0.2,
    label="Actual Data"
)

plt.plot(
    x_line,
    y_initial_line,
    linewidth=3,
    label="Before Gradient Descent"
)

plt.plot(
    x_line,
    y_final_line,
    linewidth=3,
    label="After Gradient Descent"
)

plt.xlabel("Standardized CGPA")
plt.ylabel("Salary Package")
plt.title("Linear Regression Before and After Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost Reduction")
plt.grid(True)
plt.show()
simple_lr = LinearRegression()

simple_lr.fit(
    X_simple_scaled.reshape(-1, 1),
    y_simple
)

print("Gradient Descent Weight :", round(w, 4))
print("Linear Regression Weight:", round(simple_lr.coef_[0], 4))

print()

print("Gradient Descent Intercept :", round(b, 4))
print("Linear Regression Intercept:", round(simple_lr.intercept_, 4))