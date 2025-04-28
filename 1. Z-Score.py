# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (adjust the path if necessary)
df_iris = pd.read_csv("Dataset\IRIS.csv")

# Assign proper column names
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Select numeric columns
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Apply Z-score normalization
df_iris_zscore = df_iris.copy()
df_iris_zscore[numeric_cols] = df_iris[numeric_cols].apply(zscore)

# --- Boxplot of Z-score normalized features ---
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_iris_zscore[numeric_cols])
plt.title("Boxplot of Z-Score Normalized IRIS Features")
plt.ylabel("Z-score")
plt.xlabel("Features")
plt.axhline(y=3, color='r', linestyle='--', label='Z = 3 (Outlier Threshold)')
plt.axhline(y=-3, color='r', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# --- Scatter plot: Sepal Length vs Sepal Width (Z-score normalized) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="sepal_length", 
    y="sepal_width", 
    hue="species", 
    data=df_iris_zscore,
    palette="Set2"
)
plt.title("Scatter Plot (Z-Score): Sepal Length vs Sepal Width")
plt.xlabel("Z-score: Sepal Length")
plt.ylabel("Z-score: Sepal Width")
plt.axhline(y=3, color='red', linestyle='--', linewidth=1)
plt.axhline(y=-3, color='red', linestyle='--', linewidth=1)
plt.axvline(x=3, color='red', linestyle='--', linewidth=1)
plt.axvline(x=-3, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()
