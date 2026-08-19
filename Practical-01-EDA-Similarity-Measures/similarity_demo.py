"""
Practical 1 - Measure "closeness" between real customers three different ways.
Dataset: Mall Customers (real, 200 rows)
Run: python similarity_demo.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight"})
os.makedirs("images", exist_ok=True)

# --- Load and inspect real data ---
df = pd.read_csv("../datasets/mall_customers.csv")
print("Loaded", len(df), "real customers")
print(df.describe().round(1))

# --- Chart 1: distributions of the 3 numeric features ---
cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
colors = ["#3b82f6", "#16a34a", "#f97316"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, col, color in zip(axes, cols, colors):
    ax.hist(df[col], bins=20, color=color)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("images/01_eda_distributions.png")
plt.close()

# --- Chart 2: income vs spending, colored by gender ---
plt.figure(figsize=(7.5, 6))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", s=70)
plt.title("Income vs Spending Score")
plt.tight_layout()
plt.savefig("images/02_income_vs_spending.png")
plt.close()

# --- Three distance measures on two real customers ---
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
a, b = X[0], X[1]
print("\nCustomer A:", df.iloc[0][features].to_dict())
print("Customer B:", df.iloc[1][features].to_dict())

dist_raw = euclidean(a, b)
print(f"\nEuclidean distance:  {dist_raw:.2f}  (straight-line distance)")
print(f"Manhattan distance:  {cityblock(a, b):.2f}  (sum of feature differences)")
print(f"Cosine similarity:   {1 - cosine(a, b):.4f}  (pattern similarity, ignores size)")

# --- Prove scaling changes the answer ---
X_scaled = StandardScaler().fit_transform(X)
dist_scaled = euclidean(X_scaled[0], X_scaled[1])
print(f"\nEuclidean distance BEFORE scaling: {dist_raw:.2f}")
print(f"Euclidean distance AFTER scaling:  {dist_scaled:.2f}")
print("Income ($15k-$137k) drowns out Age (18-70) unless we scale first.")

# --- Do the 3 measures agree on nearest neighbours? ---
query = X_scaled[0]
scores = pd.DataFrame({
    "customer": range(len(X_scaled)),
    "euclidean": [euclidean(query, x) for x in X_scaled],
    "manhattan": [cityblock(query, x) for x in X_scaled],
    "cosine": [cosine(query, x) for x in X_scaled],
})
top5 = {m: scores.sort_values(m)["customer"].iloc[1:6].tolist() for m in ["euclidean", "manhattan", "cosine"]}
print("\nTop 5 closest customers to Customer 0, by each measure:")
for measure, ids in top5.items():
    print(f"  {measure}: {ids}")

overlap = len(set(top5["euclidean"]) & set(top5["cosine"]))
print(f"\nEuclidean and Cosine agree on {overlap}/5 neighbours -- different measures, different answers.")

print("\nDone. Charts saved in images/")
