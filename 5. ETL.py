# Step 1: Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 2: Extract - Load the Dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")

# Assign column names (if not auto-detected)
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print("Data Extracted Successfully!")
print(df_iris.head())

# Step 3: Transform - Data Cleaning & Preprocessing

## Handling missing values (if any)
df_iris.fillna(df_iris.mean(), inplace=True)

## Standardizing numerical features
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
scaler = StandardScaler()
df_iris[numeric_cols] = scaler.fit_transform(df_iris[numeric_cols])

## Encoding categorical column ('species')
encoder = LabelEncoder()
df_iris["species_encoded"] = encoder.fit_transform(df_iris["species"])

print("Data Transformation Complete!")
print(df_iris.head())

# Step 4: Load - Save the Transformed Data
df_iris.to_csv("Transformed_IRIS.csv", index=False)

print("Data Successfully Loaded into 'Transformed_IRIS.csv'!")