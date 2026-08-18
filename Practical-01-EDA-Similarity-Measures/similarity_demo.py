"""
Practical 1 - Look at real customer data and compare 3 ways to measure "closeness".
Run: python similarity_demo.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight"})
os.makedirs("images", exist_ok=True)

df = pd.read_csv("../datasets/mall_customers.csv")
print("Loaded", len(df), "real customers")
print(df.describe().round(1))

# Picture the data
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
axes[0].hist(df["Age"], bins=20, color="#3b82f6"); axes[0].set_title("Age")
axes[1].hist(df["Annual Income (k$)"], bins=20, color="#16a34a"); axes[1].set_title("Income ($k)")
axes[2].hist(df["Spending Score (1-100)"], bins=20, color="#f97316"); axes[2].set_title("Spending Score")
plt.tight_layout()
plt.savefig("images/01_eda_distributions.png")
plt.close()

plt.figure(figsize=(7.5, 6))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", s=70)
plt.title("Income vs Spending Score")
plt.tight_layout()
plt.savefig("images/02_income_vs_spending.png")
plt.close()

# Compare 3 customers using 3 different distance measures
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
a, b = X[0], X[1]
print("\nCustomer A:", df.iloc[0][features].to_dict())
print("Customer B:", df.iloc[1][features].to_dict())
print(f"\nEuclidean distance:  {euclidean(a, b):.2f}  (straight-line distance)")
print(f"Manhattan distance:  {cityblock(a, b):.2f}  (add up each feature's difference)")
print(f"Cosine similarity:   {1 - cosine(a, b):.4f}  (how similar their *pattern* is, ignoring size)")

# Why scaling matters
X_scaled = StandardScaler().fit_transform(X)
print(f"\nEuclidean distance BEFORE scaling: {euclidean(a, b):.2f}")
print(f"Euclidean distance AFTER scaling:  {euclidean(X_scaled[0], X_scaled[1]):.2f}")
print("Income (up to $137k) drowns out Age (up to 70) unless we scale first.")

# Do different measures pick different "nearest neighbors" for the same customer?
query = X_scaled[0]
dists = {
    "euclidean": [euclidean(query, x) for x in X_scaled],
    "manhattan": [cityblock(query, x) for x in X_scaled],
    "cosine": [cosine(query, x) for x in X_scaled],
}
results = pd.DataFrame({"customer": range(len(X_scaled)), **dists})
top5 = {k: results.sort_values(k).iloc[1:6]["customer"].tolist() for k in dists}
print("\nTop 5 closest customers to Customer 0, by each measure:")
for k, v in top5.items():
    print(f"  {k}: {v}")
overlap = len(set(top5["euclidean"]) & set(top5["cosine"]))
print(f"\nEuclidean and Cosine only agree on {overlap}/5 -- different measures give different answers.")

print("\nDone. Charts saved in images/")
