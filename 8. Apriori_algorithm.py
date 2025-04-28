import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Convert numerical features into categorical bins
df_iris["sepal_length_bin"] = pd.cut(df_iris["sepal_length"], bins=3, labels=["Short", "Medium", "Long"])
df_iris["sepal_width_bin"] = pd.cut(df_iris["sepal_width"], bins=3, labels=["Narrow", "Medium", "Wide"])
df_iris["petal_length_bin"] = pd.cut(df_iris["petal_length"], bins=3, labels=["Short", "Medium", "Long"])
df_iris["petal_width_bin"] = pd.cut(df_iris["petal_width"], bins=3, labels=["Narrow", "Medium", "Wide"])

# Select only categorical columns for analysis
df_apriori = df_iris[["sepal_length_bin", "sepal_width_bin", "petal_length_bin", "petal_width_bin", "species"]]

# Convert dataset into one-hot encoded format
df_apriori_encoded = pd.get_dummies(df_apriori)

# Find frequent itemsets using Apriori
frequent_itemsets = apriori(df_apriori_encoded, min_support=0.2, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("Association Rules:\n", rules)

# Optional: Filter interesting rules based on lift or confidence
# For example, you can filter rules with a lift greater than 2:
interesting_rules = rules[rules['lift'] > 2]
print("Interesting Rules (Lift > 2):\n", interesting_rules)
