# Importing necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")
  # Ensure the file exists in your working directory
print("Initial Data:")
print(df_iris.head())  # Display the first few rows

# Correct column names (assuming the CSV has incorrect or no headers)
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print("Old Shape:", df_iris.shape)

# IQR using numpy (no 'method' for compatibility with older versions)
Q1 = np.percentile(df_iris['sepal_length'], 25)
Q3 = np.percentile(df_iris['sepal_length'], 75)
IQR = Q3 - Q1
print("IQR:", IQR)

# Outlier Detection using IQR method on 'sepal_length'
Q1 = df_iris["sepal_length"].quantile(0.25)
Q3 = df_iris["sepal_length"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_iris = df_iris[(df_iris["sepal_length"] >= lower_bound) & (df_iris["sepal_length"] <= upper_bound)]

# Print new shape after removing outliers
print("New Shape after removing outliers:", df_iris.shape)
