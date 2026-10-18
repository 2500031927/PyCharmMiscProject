import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# =========================================================================
# SETTINGS
# =========================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42
pd.set_option("display.max_columns", None)


# =========================================================================
# PATH CONFIGURATION & DATA LOADING
# =========================================================================

# Your project folder
folder_path = r"C:\Users\HP\PyCharmMiscProject\Program"

# YOUR ACTUAL DATASET FILE NAME
file_name = "placement_predict_50k Dataset.csv"

# Create complete file path
file_path = os.path.join(folder_path, file_name)

print("Looking for dataset at:")
print(file_path)

# Check whether file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"\nDataset not found!\n"
        f"Please check that this file exists:\n{file_path}\n"
    )

# Load dataset
df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================================
# INITIAL DATA EXPLORATION
# =========================================================================

print("\n================ DATA INFO ================\n")
df.info()

print("\n================ DESCRIPTIVE STATISTICS ================\n")
print(df.describe(include="all").T)


# =========================================================================
# TARGET VARIABLE ANALYSIS
# =========================================================================

target_counts = df["PlacementStatus"].value_counts()

print("\n================ TARGET COUNTS ================\n")
print(target_counts)

print("\n================ TARGET PERCENTAGE ================\n")
print(df["PlacementStatus"].value_counts(normalize=True) * 100)

print(
    "\nPlacement rate: {:.2f}%".format(
        df["PlacementStatus"].mean() * 100
    )
)


# Target class plot
plt.figure(figsize=(5, 4))

sns.countplot(
    x="PlacementStatus",
    data=df
)

plt.title("Target Class Balance: PlacementStatus")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()


# =========================================================================
# MISSING VALUES ANALYSIS
# =========================================================================

print("\n================ MISSING VALUES ================\n")

missing = df.isnull().sum()

missing = missing[
    missing > 0
].sort_values(ascending=False)

missing_pct = (
    missing / len(df) * 100
).round(2)

missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})

print(missing_df)


# Missing value plot
if not missing_df.empty:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=missing_df.index,
        y=missing_df["Missing %"]
    )

    plt.title("Percentage of Missing Values per Column")
    plt.ylabel("% Missing")
    plt.xlabel("Column")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


print(
    "\nRows with at least one missing value: {} ({:.1f}% of the dataset)".format(
        df.isnull().any(axis=1).sum(),
        df.isnull().any(axis=1).mean() * 100
    )
)


# =========================================================================
# CHECK IMPORTANT COLUMNS
# =========================================================================

print("\n================ IMPORTANT COLUMNS ================\n")

important_columns = [
    "StudentID",
    "Salary Package",
    "CGPA_Tier",
    "IsAnomaly"
]

existing_columns = [
    col for col in important_columns
    if col in df.columns
]

print(df[existing_columns].head())


# =========================================================================
# SALARY PACKAGE ANALYSIS
# =========================================================================

if "Salary Package" in df.columns:

    print("\n================ SALARY PACKAGE BY PLACEMENT ================\n")

    print(
        df.groupby("PlacementStatus")["Salary Package"].describe()
    )

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        x="PlacementStatus",
        y="Salary Package",
        data=df
    )

    plt.title("Salary Package by Placement Status")
    plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
    plt.ylabel("Salary Package")

    plt.tight_layout()
    plt.show()


# =========================================================================
# ANOMALY ANALYSIS
# =========================================================================

if "IsAnomaly" in df.columns:

    print("\n================ ANOMALY ANALYSIS ================\n")

    anomaly_ct = (
        pd.crosstab(
            df["IsAnomaly"],
            df["PlacementStatus"],
            normalize="index"
        ) * 100
    )

    print(anomaly_ct.round(2))

    anomaly_ct.plot(
        kind="bar",
        stacked=True,
        figsize=(6, 4)
    )

    plt.title("Placement Rate by IsAnomaly Flag")
    plt.ylabel("% within group")
    plt.xlabel("IsAnomaly")

    plt.legend(
        ["Not Placed", "Placed"]
    )

    plt.tight_layout()
    plt.show()


# =========================================================================
# CREATE REDUCED DATAFRAME
# =========================================================================

df_reduced = df.copy()


# =========================================================================
# TRAIN-TEST SPLIT PREPARATION
# =========================================================================

# Student ID is kept separately
if "StudentID" in df_reduced.columns:
    student_ids = df_reduced["StudentID"]
else:
    student_ids = pd.Series(
        range(len(df_reduced)),
        index=df_reduced.index,
        name="StudentID"
    )


# Target
y = df_reduced["PlacementStatus"]


# Remove target and ID
X = df_reduced.drop(
    columns=[
        "PlacementStatus",
        "StudentID"
    ],
    errors="ignore"
)


# IMPORTANT:
# If your professor said NOT to use Salary Package and IsAnomaly
# for model training, remove them here.

X = X.drop(
    columns=[
        "Salary Package",
        "IsAnomaly"
    ],
    errors="ignore"
)


# =========================================================================
# TRAIN / TEST SPLIT
# =========================================================================

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    student_ids,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\n================ TRAIN TEST SPLIT ================\n")

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# =========================================================================
# IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# =========================================================================

num_cols = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

cat_cols = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


missing_num_cols = [
    c for c in num_cols
    if X_train[c].isnull().any()
]


print("\n================ COLUMN TYPES ================\n")

print("Numeric columns:")
print(num_cols)

print("\nCategorical columns:")
print(cat_cols)

print("\nNumeric columns with missing values:")
print(missing_num_cols)


# =========================================================================
# FINAL SUMMARY
# =========================================================================

print("\n===================================================")
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("===================================================")

print("\nOriginal dataset shape :", df.shape)
print("X_train shape          :", X_train.shape)
print("X_test shape           :", X_test.shape)
print("y_train shape          :", y_train.shape)
print("y_test shape           :", y_test.shape)

print("\nNumber of numerical columns :", len(num_cols))
print("Number of categorical columns:", len(cat_cols))
