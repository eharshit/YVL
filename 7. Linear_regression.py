import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Step 2: Selecting features (X) and target variable (y)
X = df_iris[["sepal_width", "petal_length", "petal_width"]]  # Independent variables
y = df_iris["sepal_length"]  # Dependent variable

# Optional: Normalize the features (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Initialize Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model evaluation (Mean Squared Error & R-squared Score)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 8: Visualizing Actual vs Predicted values
plt.scatter(y_test, y_pred, color="blue", label="Actual vs Predicted")
plt.xlabel("Actual Sepal Length")
plt.ylabel("Predicted Sepal Length")
plt.title("Linear Regression - Iris Dataset")
plt.legend()
plt.show()

# Step 9: Optional - Visualizing Residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Sepal Length")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Step 10: Optional - Inspect the model's intercept and coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
