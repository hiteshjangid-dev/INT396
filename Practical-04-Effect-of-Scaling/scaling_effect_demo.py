"""
Practical 4 — Investigate the influence of data preprocessing by examining
the effect of feature scaling and standardization on clustering quality
and model convergence.
Run: python scaling_effect_demo.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
                      "axes.titlesize": 13, "axes.titleweight": "bold"})
os.makedirs("images", exist_ok=True)

df = pd.read_csv("../datasets/mall_customers.csv")
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
print("### DATASET: Mall Customers -- all 3 numeric features, very different scales ###")
print(df[features].describe().round(1))
print("\nFeature ranges: Age 18-70, Income 15-137 (k$), Spending Score 1-99")
print("Income has by far the largest numeric range -- exactly the scaling problem this practical studies.")

# ---------------------------------------------------------------------------
# 1. Cluster on RAW (unscaled) features
# ---------------------------------------------------------------------------
print("\n### CLUSTERING ON RAW, UNSCALED FEATURES ###")
km_raw = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
sil_raw = silhouette_score(X, km_raw.labels_)
print(f"Silhouette score (raw features): {sil_raw:.3f}")
print(f"Converged in {km_raw.n_iter_} iterations")

# ---------------------------------------------------------------------------
# 2. Cluster on StandardScaler-scaled features
# ---------------------------------------------------------------------------
print("\n### CLUSTERING ON STANDARDIZED FEATURES (StandardScaler) ###")
X_standard = StandardScaler().fit_transform(X)
km_standard = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_standard)
sil_standard = silhouette_score(X_standard, km_standard.labels_)
print(f"Silhouette score (standardized): {sil_standard:.3f}")
print(f"Converged in {km_standard.n_iter_} iterations")

# ---------------------------------------------------------------------------
# 3. Cluster on MinMax-scaled features
# ---------------------------------------------------------------------------
print("\n### CLUSTERING ON MIN-MAX SCALED FEATURES ###")
X_minmax = MinMaxScaler().fit_transform(X)
km_minmax = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_minmax)
sil_minmax = silhouette_score(X_minmax, km_minmax.labels_)
print(f"Silhouette score (min-max scaled): {sil_minmax:.3f}")
print(f"Converged in {km_minmax.n_iter_} iterations")

# ---------------------------------------------------------------------------
# 4. Direct comparison: does Age even matter in the raw clustering?
# ---------------------------------------------------------------------------
print("\n### HOW MUCH DOES EACH FEATURE ACTUALLY INFLUENCE THE RAW CLUSTERING? ###")
feature_ranges = X.max(axis=0) - X.min(axis=0)
print("Raw feature ranges:", dict(zip(features, feature_ranges.round(1))))
print("Income's range is", round(feature_ranges[1] / feature_ranges[0], 1), "x Age's range --")
print("in Euclidean distance, Income differences numerically swamp Age differences before scaling.")

# Cluster using only Age+Spending (excluding Income) as a control, to see if raw clustering
# on all 3 features actually resembles clustering on Income+Spending (Age effectively ignored)
km_income_spending_only = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X[:, [1, 2]])
from sklearn.metrics import adjusted_rand_score
ari_raw_vs_incomespending = adjusted_rand_score(km_raw.labels_, km_income_spending_only.labels_)
km_age_spending_only = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X[:, [0, 2]])
ari_raw_vs_agespending = adjusted_rand_score(km_raw.labels_, km_age_spending_only.labels_)
print(f"\nAgreement (ARI) between raw 3-feature clustering and Income+Spending-only clustering: "
      f"{ari_raw_vs_incomespending:.3f}")
print(f"Agreement (ARI) between raw 3-feature clustering and Age+Spending-only clustering:    "
      f"{ari_raw_vs_agespending:.3f}")
print("The much higher agreement with Income+Spending confirms Age is being effectively")
print("ignored in the raw (unscaled) clustering -- exactly the scaling problem in action.")

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, labels, title, sil in [
    (axes[0], km_raw.labels_, f"Raw features (silhouette={sil_raw:.3f})", sil_raw),
    (axes[1], km_standard.labels_, f"StandardScaler (silhouette={sil_standard:.3f})", sil_standard),
    (axes[2], km_minmax.labels_, f"MinMaxScaler (silhouette={sil_minmax:.3f})", sil_minmax),
]:
    sns.scatterplot(x=df["Annual Income (k$)"], y=df["Spending Score (1-100)"], hue=labels,
                     palette="tab10", s=55, ax=ax, legend=False)
    ax.set_title(title, fontsize=11)
plt.suptitle("Effect of Feature Scaling on k-Means Clustering (Age, Income, Spending Score)",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("images/01_scaling_comparison.png")
plt.close()

plt.figure(figsize=(7, 4.3))
methods = ["Raw\n(unscaled)", "StandardScaler", "MinMaxScaler"]
scores = [sil_raw, sil_standard, sil_minmax]
plt.bar(methods, scores, color=["#dc2626", "#16a34a", "#2563eb"])
plt.title("Silhouette Score by Scaling Method")
plt.ylabel("Silhouette score")
for i, s in enumerate(scores):
    plt.text(i, s + 0.01, f"{s:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("images/02_silhouette_by_method.png")
plt.close()

print("\nAll charts saved to ./images/")
