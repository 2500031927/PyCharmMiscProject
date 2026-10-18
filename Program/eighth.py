import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv("placement_predict_50k Dataset.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# Input feature
# ==========================================

X = df[["CGPA"]]

# Target
y = df["PlacementStatus"]

print("\nInput:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nTarget Value Counts:")
print(y.value_counts())

print("\nUnique Target Values:")
print(y.unique())


# ==========================================
# Check for missing values
# ==========================================

print("\nMissing values in CGPA:")
print(X.isnull().sum())

print("\nMissing values in PlacementStatus:")
print(y.isnull().sum())


# ==========================================
# Remove rows with missing values
# ==========================================

data = pd.concat([X, y], axis=1)

data = data.dropna()

X = data[["CGPA"]]
y = data["PlacementStatus"]


print("\nAfter removing missing values:")
print("Input samples:", len(X))
print("Target samples:", len(y))


# ==========================================
# Plot CGPA vs Placement Status
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X["CGPA"],
    y,
    alpha=0.3
)

plt.xlabel("CGPA")
plt.ylabel("Placement Status")
plt.title("CGPA vs Placement Status")

plt.yticks(
    [0, 1],
    ["Not Placed", "Placed"]
)

plt.grid(True, alpha=0.2)

plt.show()


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# Create Logistic Regression Model
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)


# ==========================================
# Predict probabilities
# ==========================================

probabilities = model.predict_proba(X_test)

print("\nFirst 10 Prediction Probabilities:")
print(probabilities[:10])


# ==========================================
# Predict classes
# ==========================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])


# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Placed", "Placed"]
    )
)






y_pred = model.predict(X_test.values.reshape(-1, 1))

print(y_pred[:20])
y_pred = model.predict(X_test.values.reshape(-1, 1))

print(y_pred[:20])
cm = confusion_matrix(y_test, y_pred)

print(cm)
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks([0, 1], ["Not Placed", "Placed"])
plt.yticks([0, 1], ["Not Placed", "Placed"])

plt.colorbar()

plt.show()
print(classification_report(y_test, y_pred))