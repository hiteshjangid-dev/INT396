# Practical 3 — k-Means From Scratch and Validation Against Scikit-learn
# Google Colab / Python friendly
#
# Dataset: Mall_Customers.csv
# Required columns:
#   Annual Income (k$)
#   Spending Score (1-100)

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ============================================================
# 1. LOAD DATA
# ============================================================

# Google Colab:
# from google.colab import files
# files.upload()

DATA_FILE = "Mall_Customers.csv"
K = 5
RANDOM_STATE = 42

df = pd.read_csv(DATA_FILE)

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].dropna().to_numpy(dtype=float)

print("Dataset shape:", X.shape)
print(df.head())


# ============================================================
# 2. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 3. k-MEANS FROM SCRATCH
# ============================================================

def kmeans_scratch(X, k=5, max_iter=100, tol=1e-4, random_state=42):
    """Small, readable k-Means implementation using NumPy."""

    rng = np.random.default_rng(random_state)

    # ---------- k-Means++ initialization ----------
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(len(X))]

    for i in range(1, k):
        dist_sq = ((X[:, None, :] - centroids[None, :i, :]) ** 2).sum(axis=2)
        min_dist_sq = dist_sq.min(axis=1)

        total = min_dist_sq.sum()

        # Safe fallback for identical points
        if total == 0:
            centroids[i] = X[rng.integers(len(X))]
        else:
            probabilities = min_dist_sq / total
            centroids[i] = X[rng.choice(len(X), p=probabilities)]

    history = []

    # ---------- Lloyd's algorithm ----------
    for iteration in range(max_iter):

        # 1. Distance + cluster assignment
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        # 2. WCSS / inertia
        inertia = distances[np.arange(len(X)), labels].sum()
        history.append(inertia)

        # 3. Recalculate centroids
        new_centroids = centroids.copy()

        for i in range(k):
            points = X[labels == i]
            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)

        # 4. Convergence check
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    return labels, centroids, history


# ============================================================
# 4. RUN FROM-SCRATCH MODEL
# ============================================================

start = time.perf_counter()

scratch_labels, scratch_centroids, history = kmeans_scratch(
    X_scaled,
    k=K,
    random_state=RANDOM_STATE
)

scratch_time = time.perf_counter() - start


# ============================================================
# 5. RUN SCIKIT-LEARN MODEL
# ============================================================

start = time.perf_counter()

sklearn_model = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=1,                 # fair single-run comparison
    random_state=RANDOM_STATE
)

sklearn_labels = sklearn_model.fit_predict(X_scaled)

sklearn_time = time.perf_counter() - start


# ============================================================
# 6. VALIDATION
# ============================================================

scratch_inertia = history[-1]
sklearn_inertia = sklearn_model.inertia_

scratch_silhouette = silhouette_score(X_scaled, scratch_labels)
sklearn_silhouette = silhouette_score(X_scaled, sklearn_labels)

ari = adjusted_rand_score(scratch_labels, sklearn_labels)

print("\n========== VALIDATION RESULTS ==========")
print(f"Scratch inertia        : {scratch_inertia:.4f}")
print(f"Scikit-learn inertia   : {sklearn_inertia:.4f}")
print(f"Scratch silhouette     : {scratch_silhouette:.4f}")
print(f"Scikit-learn silhouette: {sklearn_silhouette:.4f}")
print(f"ARI                    : {ari:.4f}")
print(f"Scratch iterations     : {len(history)}")
print(f"Scratch time           : {scratch_time * 1000:.2f} ms")
print(f"Scikit-learn time      : {sklearn_time * 1000:.2f} ms")


# ============================================================
# 7. CREATE IMAGES FOLDER
# ============================================================

os.makedirs("images", exist_ok=True)


# ============================================================
# 8. VISUAL 1 — CONVERGENCE CURVE
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(history) + 1),
    history,
    marker="o",
    linewidth=2
)
plt.xlabel("Iteration")
plt.ylabel("Inertia (WCSS)")
plt.title("k-Means Convergence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/01_convergence_curve.png", dpi=150)
plt.show()


# ============================================================
# 9. VISUAL 2 — SCRATCH VS SCIKIT-LEARN
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# From scratch
ax[0].scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=scratch_labels,
    cmap="viridis",
    s=45
)
ax[0].scatter(
    scratch_centroids[:, 0],
    scratch_centroids[:, 1],
    marker="X",
    s=220,
    c="red",
    edgecolors="black",
    linewidths=1.2,
    label="Centroids"
)
ax[0].set_title("k-Means From Scratch")
ax[0].set_xlabel("Annual Income (standardized)")
ax[0].set_ylabel("Spending Score (standardized)")
ax[0].legend()

# Scikit-learn
ax[1].scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=sklearn_labels,
    cmap="viridis",
    s=45
)
ax[1].scatter(
    sklearn_model.cluster_centers_[:, 0],
    sklearn_model.cluster_centers_[:, 1],
    marker="X",
    s=220,
    c="red",
    edgecolors="black",
    linewidths=1.2,
    label="Centroids"
)
ax[1].set_title("Scikit-learn k-Means")
ax[1].set_xlabel("Annual Income (standardized)")
ax[1].set_ylabel("Spending Score (standardized)")
ax[1].legend()

plt.suptitle("From Scratch vs Scikit-learn", fontsize=14)
plt.tight_layout()
plt.savefig("images/02_scratch_vs_sklearn.png", dpi=150)
plt.show()


# ============================================================
# 10. VISUAL 3 — SCALABILITY COMPARISON
# ============================================================

sizes = [50, 100, 150, 200, 500, 1000]
scratch_times = []
sklearn_times = []

# Repeat the real data to create larger test sets.
# This is only a classroom timing demonstration.
for n in sizes:
    X_test = np.resize(X_scaled, (n, X_scaled.shape[1]))

    start = time.perf_counter()
    kmeans_scratch(
        X_test,
        k=K,
        random_state=RANDOM_STATE
    )
    scratch_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=1,
        random_state=RANDOM_STATE
    ).fit(X_test)
    sklearn_times.append(time.perf_counter() - start)

plt.figure(figsize=(8, 5))
plt.plot(
    sizes,
    np.array(scratch_times) * 1000,
    marker="o",
    linewidth=2,
    label="From scratch"
)
plt.plot(
    sizes,
    np.array(sklearn_times) * 1000,
    marker="o",
    linewidth=2,
    label="Scikit-learn"
)
plt.xlabel("Number of samples")
plt.ylabel("Execution Time (ms)")
plt.title("Scalability Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/03_scalability_comparison.png", dpi=150)
plt.show()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n========== PRACTICAL SUMMARY ==========")
print("1. k-Means was implemented from scratch.")
print("2. k-Means++ was used for initialization.")
print("3. Features were standardized before clustering.")
print("4. The result was validated against Scikit-learn.")
print("5. Convergence was visualized.")
print("6. Scalability was demonstrated.")
print("\nGenerated files:")
print("images/01_convergence_curve.png")
print("images/02_scratch_vs_sklearn.png")
print("images/03_scalability_comparison.png")
