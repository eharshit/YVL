import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Select features for clustering (excluding 'species')
X = df_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

# Standardize the features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_iris["cluster"] = kmeans.fit_predict(X_scaled)

# Cluster Centroids
centroids = kmeans.cluster_centers_

# Visualizing the clusters using petal length vs petal width
plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=df_iris["cluster"], cmap="viridis", label="Clusters")
plt.scatter(centroids[:, 2], centroids[:, 3], c="red", marker="X", s=200, label="Centroids")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering - Iris Dataset")
plt.legend()
plt.show()