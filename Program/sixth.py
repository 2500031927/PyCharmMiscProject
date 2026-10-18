import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("Libraries imported successfully.")

# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = "placement_predict_50k Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# -----------------------------
# MISSING VALUE CHECK
# -----------------------------

print("\nMISSING VALUE CHECK")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentage:")
print(missing_percentage)


# -----------------------------
# DUPLICATE CHECK
# -----------------------------

duplicate_count = df.duplicated().sum()

print("\nDUPLICATE CHECK")

print("Number of duplicate rows:", duplicate_count)


# -----------------------------
# DEFINE INPUT AND TARGET
# -----------------------------

# Target variable
y = df["Salary Package"]

# Input features
X = df.drop(
    ["Salary Package", "PlacementStatus", "IsAnomaly"],
    axis=1
)

# Remove StudentID
if "StudentID" in X.columns:
    X = X.drop("StudentID", axis=1)

print("\nInput columns after removing StudentID:")
print(X.columns.tolist())

print("\nTarget column:")
print("Salary Package")


# -----------------------------
# IDENTIFY COLUMNS
# -----------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCOLUMNS")

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# -----------------------------
# NUMERICAL PREPROCESSING
# -----------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

print("\nNumerical preprocessing pipeline created.")


# -----------------------------
# CATEGORICAL PREPROCESSING
# -----------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

print("Categorical preprocessing pipeline created.")


# -----------------------------
# COLUMN TRANSFORMER
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

print("Preprocessor created successfully.")


# -----------------------------
# CREATE MODEL
# -----------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

print("Multiple Linear Regression model created.")


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN TEST SPLIT")

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# -----------------------------
# TRAIN MODEL
# -----------------------------

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully.")


# -----------------------------
# PREDICTION
# -----------------------------

y_pred = model.predict(X_test)

print("\nPredictions generated successfully.")


# -----------------------------
# MODEL EVALUATION
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)

results = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\n ACTUAL VS PREDICTED ")

print(results.head(20))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid(True)
plt.show()