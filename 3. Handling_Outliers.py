# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df_iris = pd.read_csv(r"Dataset\IRIS.csv")
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Select numeric columns
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Step 1: Fill missing values using column mean
df_iris[numeric_cols] = df_iris[numeric_cols].fillna(df_iris[numeric_cols].mean())

# Step 2: Remove outliers using Standard Deviation method
df_cleaned = df_iris.copy()

for col in numeric_cols:
    mean = df_cleaned[col].mean()
    std = df_cleaned[col].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

# Show shapes before and after
print("Original shape:", df_iris.shape)
print("After removing outliers:", df_cleaned.shape)

# Optional: Boxplot after removing outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned[numeric_cols])
plt.title("Boxplot After Removing Outliers (Std Dev Method)")
plt.tight_layout()
plt.show()
