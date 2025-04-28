import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
df_iris = pd.read_csv("Dataset\IRIS.csv")

# Assign column names (if not auto-detected)
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Step 2: Check for duplicates and drop them
df_iris.drop_duplicates(inplace=True)

# Step 3: Convert numerical features into categorical bins
df_iris["sepal_length_bin"] = pd.cut(df_iris["sepal_length"], bins=3, labels=["Short", "Medium", "Long"])
df_iris["sepal_width_bin"] = pd.cut(df_iris["sepal_width"], bins=3, labels=["Narrow", "Medium", "Wide"])
df_iris["petal_length_bin"] = pd.cut(df_iris["petal_length"], bins=3, labels=["Short", "Medium", "Long"])
df_iris["petal_width_bin"] = pd.cut(df_iris["petal_width"], bins=3, labels=["Narrow", "Medium", "Wide"])

# Step 4: Select only categorical columns for analysis (excluding 'species' from the initial transformation)
df_apriori = df_iris[["sepal_length_bin", "sepal_width_bin", "petal_length_bin", "petal_width_bin"]]

# Step 5: Convert the dataset into one-hot encoded format
df_apriori_encoded = pd.get_dummies(df_apriori)

# Step 6: Find frequent itemsets using Apriori (with min_support of 0.2)
frequent_itemsets = apriori(df_apriori_encoded, min_support=0.2, use_colnames=True)

# Step 7: Print the frequent itemsets
print("Frequent Itemsets:\n", frequent_itemsets)

# Step 8: Generate association rules with 'lift' as the metric (min_threshold = 1)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Step 9: Print the association rules
print("Association Rules:\n", rules)

# Step 10: Optionally, filter the association rules based on certain thresholds (for example, support and lift)
rules_filtered = rules[(rules['lift'] >= 1) & (rules['support'] >= 0.2)]

print("\nFiltered Association Rules (Lift >= 1 & Support >= 0.2):\n", rules_filtered)
