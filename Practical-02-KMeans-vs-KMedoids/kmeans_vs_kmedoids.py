"""
Practical 2 - Group real customers using k-Means and k-Medoids, then compare.
Run: python kmeans_vs_kmedoids.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight"})
os.makedirs("images", exist_ok=True)

df = pd.read_csv("../datasets/mall_customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
X_scaled = StandardScaler().fit_transform(X)
print("Loaded", len(df), "real customers")


def kmedoids(X, k, seed=42, max_iter=100):
    """k-Medoids: like k-Means, but the center of each group must be a REAL point,
    not an average. More resistant to weird outlier points."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    medoids = rng.choice(n, k, replace=False)
    dist = cdist(X, X)
    cost = dist[:, medoids].min(axis=1).sum()
    for _ in range(max_iter):
        improved = False
        for i in range(k):
            for candidate in range(n):
                if candidate in medoids:
                    continue
                trial = medoids.copy()
                trial[i] = candidate
                new_cost = dist[:, trial].min(axis=1).sum()
                if new_cost < cost:
                    medoids, cost = trial, new_cost
                    improved = True
        if not improved:
            break
    labels = dist[:, medoids].argmin(axis=1)
    return labels, medoids


# k-Means
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_scaled)
print("k-Means silhouette score:", round(silhouette_score(X_scaled, km.labels_), 3))

# k-Medoids
med_labels, med_idx = kmedoids(X_scaled, k=5)
print("k-Medoids silhouette score:", round(silhouette_score(X_scaled, med_labels), 3))
print("\nReal customers chosen as group centers (medoids):")
print(df.iloc[med_idx][["CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)"]])

# Test: what happens if we add one weird outlier customer?
X_outlier = np.vstack([X, [[45, 250]]])
X_outlier_scaled = StandardScaler().fit_transform(X_outlier)
km_outlier = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_outlier_scaled)
shift = np.linalg.norm(km.cluster_centers_ - km_outlier.cluster_centers_[:5], axis=1).mean()
print(f"\nOne weird customer shifted k-Means' group centers by {shift:.2f} on average.")
print("k-Medoids can't be pulled like this -- its centers are always real points.")

# Picture both results side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=km.labels_, palette="tab10", s=60, ax=axes[0], legend=False)
axes[0].set_title(f"k-Means (score={silhouette_score(X_scaled, km.labels_):.3f})")
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=med_labels, palette="tab10", s=60, ax=axes[1], legend=False)
axes[1].scatter(X[med_idx, 0], X[med_idx, 1], c="black", marker="D", s=180, label="Real medoid customers")
axes[1].set_title(f"k-Medoids (score={silhouette_score(X_scaled, med_labels):.3f})")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.savefig("images/01_kmeans_vs_kmedoids.png")
plt.close()

print("\nDone. Chart saved in images/")
