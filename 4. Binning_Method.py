import pandas as pd
import numpy as np

# Load the dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")

# Assign column names
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Check for any missing values
print("Missing values:", df_iris.isnull().sum())

# Define the number of bins
num_bins = 3  # You can change this as needed

# Create bin labels
bin_labels = ["Small", "Medium", "Large"]

# Perform equal-width binning on sepal_length
df_iris["sepal_length_binned"] = pd.cut(df_iris["sepal_length"], bins=num_bins, labels=bin_labels)

# Show the first few rows of the dataset
print(df_iris.head())
