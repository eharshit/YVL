# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load & Preprocess Data
df_iris = pd.read_csv("Dataset\IRIS.csv")
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Encode the target variable (species)
encoder = LabelEncoder()
df_iris["species_encoded"] = encoder.fit_transform(df_iris["species"])

# Define features (X) and target variable (y)
X = df_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df_iris["species_encoded"]

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Classification Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 6: Visualizing Confusion Matrix
plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()