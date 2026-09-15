"""
Practical 2 — Develop a customer segmentation solution by implementing and
comparing k-Means and k-Medoids clustering techniques on the Mall Customers
dataset.
Run: python kmeans_vs_kmedoids.py
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
                      "axes.titlesize": 13, "axes.titleweight": "bold"})
os.makedirs("images", exist_ok=True)

df = pd.read_csv("../datasets/mall_customers.csv")
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
X_scaled = StandardScaler().fit_transform(X)
print("### DATASET: Mall Customers -- Income vs Spending Score ###")
print("Shape:", X.shape)


def kmedoids_pam(X, k, random_state=42, max_iter=100):
    """A from-scratch implementation of k-Medoids using the classic PAM
    (Partitioning Around Medoids) swap strategy: unlike k-Means, every
    cluster center is a REAL data point (a medoid), not an averaged,
    possibly non-existent point -- more robust to outliers."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    medoid_idx = rng.choice(n, k, replace=False)
    distances = cdist(X, X, metric="euclidean")

    def total_cost(medoids):
        return distances[:, medoids].min(axis=1).sum()

    current_cost = total_cost(medoid_idx)
    for _ in range(max_iter):
        improved = False
        for m_pos in range(k):
            for candidate in range(n):
                if candidate in medoid_idx:
                    continue
                new_medoids = medoid_idx.copy()
                new_medoids[m_pos] = candidate
                new_cost = total_cost(new_medoids)
                if new_cost < current_cost:
                    medoid_idx = new_medoids
                    current_cost = new_cost
                    improved = True
        if not improved:
            break
    labels = distances[:, medoid_idx].argmin(axis=1)
    return labels, medoid_idx, current_cost


# ---------------------------------------------------------------------------
# 1. k-Means
# ---------------------------------------------------------------------------
print("\n### K-MEANS ###")
start = time.time()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_time = time.time() - start
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette score: {kmeans_sil:.3f}")
print(f"Time taken: {kmeans_time*1000:.1f} ms")
print("Cluster sizes:", np.bincount(kmeans_labels))

# ---------------------------------------------------------------------------
# 2. k-Medoids (from scratch, PAM)
# ---------------------------------------------------------------------------
print("\n### K-MEDOIDS (PAM, implemented from scratch) ###")
start = time.time()
kmedoids_labels, medoid_idx, medoid_cost = kmedoids_pam(X_scaled, k=5, random_state=42)
kmedoids_time = time.time() - start
kmedoids_sil = silhouette_score(X_scaled, kmedoids_labels)
print(f"Silhouette score: {kmedoids_sil:.3f}")
print(f"Time taken: {kmedoids_time*1000:.1f} ms")
print("Cluster sizes:", np.bincount(kmedoids_labels))
print("\nReal customers chosen as medoids (actual data points, not averages):")
print(df.iloc[medoid_idx][["CustomerID", "Age"] + features])

# ---------------------------------------------------------------------------
# 3. Robustness to outliers: inject one real-scale extreme customer
# ---------------------------------------------------------------------------
print("\n### ROBUSTNESS TEST: adding one extreme outlier customer ###")
X_with_outlier = np.vstack([X, [[45, 250]]])  # a hypothetical ultra-high-income customer
X_with_outlier_scaled = StandardScaler().fit_transform(X_with_outlier)

km_outlier = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_with_outlier_scaled)
kmed_outlier_labels, kmed_outlier_medoids, _ = kmedoids_pam(X_with_outlier_scaled, k=5, random_state=42)

km_centroid_shift = np.linalg.norm(kmeans.cluster_centers_ - km_outlier.cluster_centers_[:5], axis=1).mean()
print(f"Average k-Means centroid shift caused by 1 outlier (out of 201 points): {km_centroid_shift:.3f}")
print("k-Medoids centers are always real data points, so a single extreme outlier can only ever")
print("become a medoid itself (isolated in its own cluster) -- it cannot silently drag an averaged")
print("centroid away from the bulk of real customers the way a k-Means mean can.")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette="tab10", s=60, ax=axes[0], legend=False)
centroids_original = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
axes[0].scatter(centroids_original[:, 0], centroids_original[:, 1], color="black", marker="X", s=250,
                 label="Centroids (averaged points)", edgecolor="white", linewidth=1.5)
axes[0].set_title(f"k-Means (silhouette={kmeans_sil:.3f})")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1-100)")
axes[0].legend()

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmedoids_labels, palette="tab10", s=60, ax=axes[1], legend=False)
axes[1].scatter(X[medoid_idx, 0], X[medoid_idx, 1], color="black", marker="D", s=220,
                 label="Medoids (real customers)", edgecolor="white", linewidth=1.5)
axes[1].set_title(f"k-Medoids / PAM (silhouette={kmedoids_sil:.3f})")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score (1-100)")
axes[1].legend()

plt.suptitle("k-Means vs k-Medoids: Real Mall Customer Segmentation", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("images/01_kmeans_vs_kmedoids.png")
plt.close()

print("\nAll charts saved to ./images/")
