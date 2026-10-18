import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("placement_predict_50k Dataset.csv")
print("Dataset Loaded Successfully")
df.head()
df.tail()
print(df.shape)
print(df.columns.tolist())
df.info()
print(df.dtypes)
df.describe(include='all')
print(df.isnull().sum())
print((df.isnull().sum()/len(df))*100)
mean_df = df.copy()
numeric_columns = [
    'Workshops',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'CodingTestScore',
    'MockInterviewScore'
]

mean_df[numeric_columns] = mean_df[numeric_columns].fillna(
    mean_df[numeric_columns].mean()
)

mean_df.head(10)
comparison = pd.DataFrame({
    "Original": df["Workshops"],
    "After Mean": mean_df["Workshops"]
})

comparison.head(20)
median_df = df.copy()
median_df[numeric_columns] = median_df[numeric_columns].fillna(
    median_df[numeric_columns].median()
)

comparison = pd.DataFrame({
    "Original": df["CodingTestScore"],
    "Median": median_df["CodingTestScore"]
})

comparison.head(20)
mode_df = df.copy()

for col in numeric_columns:
    mode_df[col] = mode_df[col].fillna(mode_df[col].mode()[0])

# Create a copy of only the rows with missing values
missing_rows = df[df["MockInterviewScore"].isnull()].copy()

# Add the filled values from the mode-imputed dataset
missing_rows["Filled_With_Mode"] = mode_df.loc[
    missing_rows.index,
    "MockInterviewScore"
]

# Display only these two columns
missing_rows[["MockInterviewScore", "Filled_With_Mode"]]
print("Original Dataset")
print(df[numeric_columns].isnull().sum())
print("\nAfter Mean Imputation")
print(mean_df[numeric_columns].isnull().sum())

print("\nAfter Median Imputation")
print(median_df[numeric_columns].isnull().sum())

print("\nAfter Mode Imputation")
print(mode_df[numeric_columns].isnull().sum())
comparison = pd.DataFrame({
    "Original": df["MockInterviewScore"],
    "After Mean": mean_df["MockInterviewScore"],
    "After Median": median_df["MockInterviewScore"],
    "After Mode": mode_df["MockInterviewScore"]
})

comparison[df["MockInterviewScore"].isnull()]
temp=df.dropna()
print(temp.shape)
print(df.corr(numeric_only=True))
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Set up the matplotlib figure size
plt.figure(figsize=(10, 8))

# Draw the heatmap
sns.heatmap(
    corr_matrix,
    annot=True,  # Shows the correlation numbers in the boxes
    cmap="coolwarm",  # Color palette (blue for negative, red for positive)
    fmt=".2f",  # Rounds numbers to 2 decimal places
    linewidths=0.5,  # Adds thin lines between the squares
    vmin=-1,  # Sets the minimum value for the color scale
    vmax=1,  # Sets the maximum value for the color scale
)

# Display the plot
plt.show()
num=df.select_dtypes(include='number').columns
df[num[0]].hist()
plt.show()
df.to_csv("PlacementPredict_Cleaned.csv",index=False)
print("\nCleaned Dataset Saved Successfully")
print("File Name : PlacementPredict_Cleaned.csv")


