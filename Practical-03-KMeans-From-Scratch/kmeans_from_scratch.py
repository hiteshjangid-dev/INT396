"""
Practical 3 — Design a scalable clustering model by implementing k-Means
from scratch and validating its performance using the Scikit-learn
framework.
Run: python kmeans_from_scratch.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
                      "axes.titlesize": 13, "axes.titleweight": "bold"})
os.makedirs("images", exist_ok=True)

df = pd.read_csv("../datasets/mall_customers.csv")
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
X_scaled = StandardScaler().fit_transform(X)


class KMeansFromScratch:
    """A from-scratch implementation of Lloyd's k-Means algorithm:
    1. Initialize k centroids (k-Means++ initialization, same strategy scikit-learn defaults to)
    2. Assign every point to its nearest centroid
    3. Recompute each centroid as the mean of its assigned points
    4. Repeat until centroids stop moving (convergence) or max_iter is reached
    """

    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeans_plusplus_init(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[rng.integers(n_samples)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def fit(self, X):
        self.centroids_ = self._kmeans_plusplus_init(X)
        self.n_iter_ = 0
        self.inertia_history_ = []
        for iteration in range(self.max_iter):
            distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
            labels = distances.argmin(axis=1)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids_[k]
                for k in range(self.n_clusters)
            ])
            inertia = sum(np.sum((X[labels == k] - new_centroids[k]) ** 2) for k in range(self.n_clusters))
            self.inertia_history_.append(inertia)
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.n_iter_ = iteration + 1
            if shift < self.tol:
                break
        self.labels_ = labels
        self.inertia_ = self.inertia_history_[-1]
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return distances.argmin(axis=1)


print("### K-MEANS FROM SCRATCH vs SCIKIT-LEARN, on real Mall Customers data ###")

# ---------------------------------------------------------------------------
# 1. Run both implementations
# ---------------------------------------------------------------------------
scratch = KMeansFromScratch(n_clusters=5, random_state=42).fit(X_scaled)
sklearn_km = KMeans(n_clusters=5, random_state=42, n_init=1, init="k-means++").fit(X_scaled)

print(f"\nFrom-scratch implementation converged in {scratch.n_iter_} iterations")
print(f"From-scratch inertia:    {scratch.inertia_:.3f}")
print(f"scikit-learn inertia:    {sklearn_km.inertia_:.3f}")

scratch_sil = silhouette_score(X_scaled, scratch.labels_)
sklearn_sil = silhouette_score(X_scaled, sklearn_km.labels_)
print(f"\nFrom-scratch silhouette score: {scratch_sil:.4f}")
print(f"scikit-learn silhouette score: {sklearn_sil:.4f}")

ari = adjusted_rand_score(scratch.labels_, sklearn_km.labels_)
print(f"\nAdjusted Rand Index between the two labelings: {ari:.3f} (1.0 = identical clustering)")

# ---------------------------------------------------------------------------
# 2. Convergence curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(7.5, 4.6))
plt.plot(range(1, len(scratch.inertia_history_) + 1), scratch.inertia_history_, marker="o", color="#2563eb")
plt.title("From-Scratch k-Means: Inertia Decreasing Toward Convergence")
plt.xlabel("Iteration")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.tight_layout()
plt.savefig("images/01_convergence_curve.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. Side-by-side result comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=scratch.labels_, palette="tab10", s=60, ax=axes[0], legend=False)
centroids_original = StandardScaler().fit(X).inverse_transform(scratch.centroids_)
axes[0].scatter(centroids_original[:, 0], centroids_original[:, 1], color="black", marker="X", s=250,
                 edgecolor="white", linewidth=1.5)
axes[0].set_title(f"From-Scratch k-Means (silhouette={scratch_sil:.3f})")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1-100)")

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=sklearn_km.labels_, palette="tab10", s=60, ax=axes[1], legend=False)
centroids_sk_original = StandardScaler().fit(X).inverse_transform(sklearn_km.cluster_centers_)
axes[1].scatter(centroids_sk_original[:, 0], centroids_sk_original[:, 1], color="black", marker="X", s=250,
                 edgecolor="white", linewidth=1.5)
axes[1].set_title(f"scikit-learn KMeans (silhouette={sklearn_sil:.3f})")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score (1-100)")

plt.suptitle(f"Validation: From-Scratch vs scikit-learn (Adjusted Rand Index = {ari:.2f})",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("images/02_scratch_vs_sklearn.png")
plt.close()

# ---------------------------------------------------------------------------
# 4. Timing comparison (scalability discussion)
# ---------------------------------------------------------------------------
import time
sizes = [50, 100, 150, 200]
scratch_times, sklearn_times = [], []
for n in sizes:
    Xs = X_scaled[:n]
    t0 = time.time()
    KMeansFromScratch(n_clusters=5, random_state=42).fit(Xs)
    scratch_times.append((time.time() - t0) * 1000)
    t0 = time.time()
    KMeans(n_clusters=5, random_state=42, n_init=1).fit(Xs)
    sklearn_times.append((time.time() - t0) * 1000)

print("\n### SCALABILITY: Vectorized Python loop vs scikit-learn's optimized C implementation ###")
for n, st, kt in zip(sizes, scratch_times, sklearn_times):
    print(f"  n={n}: from-scratch={st:.2f}ms | scikit-learn={kt:.2f}ms")

plt.figure(figsize=(7.5, 4.6))
plt.plot(sizes, scratch_times, marker="o", label="From-scratch implementation", color="#dc2626")
plt.plot(sizes, sklearn_times, marker="o", label="scikit-learn (optimized)", color="#16a34a")
plt.title("Scalability: From-Scratch vs Production-Grade Implementation")
plt.xlabel("Number of real customers (n)")
plt.ylabel("Time (ms)")
plt.legend()
plt.tight_layout()
plt.savefig("images/03_scalability_comparison.png")
plt.close()

print("\nAll charts saved to ./images/")
