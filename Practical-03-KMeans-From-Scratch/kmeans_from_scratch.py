# Practical 3 — k-Means From Scratch and Validation with Scikit-learn
# Google Colab friendly
#
# Dataset: Mall_Customers.csv
# Required columns:
#   Annual Income (k$)
#   Spending Score (1-100)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

# In Google Colab, upload Mall_Customers.csv using:
# from google.colab import files
# files.upload()

df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

print("Dataset shape:", X.shape)
print(df.head())


# ============================================================
# 2. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 5


# ============================================================
# 3. k-MEANS FROM SCRATCH
# ============================================================

def kmeans_scratch(X, k=5, max_iter=100, tol=1e-4, random_state=42):
    rng = np.random.default_rng(random_state)

    # k-Means++ initialization
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(len(X))]

    for i in range(1, k):
        dist_sq = np.min(
            ((X[:, None, :] - centroids[None, :i, :]) ** 2).sum(axis=2),
            axis=1
        )
        probabilities = dist_sq / dist_sq.sum()
        centroids[i] = X[rng.choice(len(X), p=probabilities)]

    history = []

    for iteration in range(max_iter):

        # Assign each point to nearest centroid
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        # Calculate WCSS / inertia
        inertia = distances[np.arange(len(X)), labels].sum()
        history.append(inertia)

        # Recalculate centroids
        new_centroids = centroids.copy()

        for i in range(k):
            points = X[labels == i]
            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)

        # Stop when centroids barely move
        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            break

    return labels, centroids, history


start = time.perf_counter()

scratch_labels, scratch_centroids, history = kmeans_scratch(
    X_scaled, k=k, random_state=42
)

scratch_time = time.perf_counter() - start


# ============================================================
# 4. VALIDATE WITH SCIKIT-LEARN
# ============================================================

start = time.perf_counter()

sklearn_model = KMeans(
    n_clusters=k,
    init="k-means++",
    n_init=1,
    random_state=42
)

sklearn_labels = sklearn_model.fit_predict(X_scaled)

sklearn_time = time.perf_counter() - start


# ============================================================
# 5. EVALUATION
# ============================================================

scratch_inertia = history[-1]
sklearn_inertia = sklearn_model.inertia_

scratch_silhouette = silhouette_score(X_scaled, scratch_labels)
sklearn_silhouette = silhouette_score(X_scaled, sklearn_labels)

ari = adjusted_rand_score(scratch_labels, sklearn_labels)

print("\n========== RESULTS ==========")
print(f"Scratch inertia       : {scratch_inertia:.4f}")
print(f"Scikit-learn inertia   : {sklearn_inertia:.4f}")
print(f"Scratch silhouette     : {scratch_silhouette:.4f}")
print(f"Scikit-learn silhouette: {sklearn_silhouette:.4f}")
print(f"ARI (label agreement)  : {ari:.4f}")
print(f"Scratch iterations     : {len(history)}")
print(f"Scratch time           : {scratch_time*1000:.2f} ms")
print(f"Scikit-learn time      : {sklearn_time*1000:.2f} ms")


# ============================================================
# 6. CONVERGENCE CURVE
# ============================================================

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(history) + 1), history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Inertia (WCSS)")
plt.title("k-Means Convergence")
plt.grid(True)
plt.show()


# ============================================================
# 7. SCRATCH vs SCIKIT-LEARN
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=scratch_labels, cmap="tab10")
ax[0].scatter(
    scratch_centroids[:, 0], scratch_centroids[:, 1],
    marker="X", s=180, color="black", label="Centroids"
)
ax[0].set_title("k-Means From Scratch")
ax[0].set_xlabel("Annual Income (standardized)")
ax[0].set_ylabel("Spending Score (standardized)")
ax[0].legend()

ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=sklearn_labels, cmap="tab10")
ax[1].scatter(
    sklearn_model.cluster_centers_[:, 0],
    sklearn_model.cluster_centers_[:, 1],
    marker="X", s=180, color="black", label="Centroids"
)
ax[1].set_title("Scikit-learn k-Means")
ax[1].set_xlabel("Annual Income (standardized)")
ax[1].set_ylabel("Spending Score (standardized)")
ax[1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# 8. OPTIONAL SCALABILITY DEMO
# ============================================================

sizes = [50, 100, 150, 200, 500, 1000]
scratch_times = []
sklearn_times = []

for n in sizes:
    X_test = np.tile(X_scaled, (n // len(X_scaled) + 1, 1))[:n]

    start = time.perf_counter()
    kmeans_scratch(X_test, k=5, random_state=42)
    scratch_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    KMeans(n_clusters=5, n_init=1, random_state=42).fit(X_test)
    sklearn_times.append(time.perf_counter() - start)

plt.figure(figsize=(7, 4))
plt.plot(sizes, np.array(scratch_times) * 1000, marker="o", label="From scratch")
plt.plot(sizes, np.array(sklearn_times) * 1000, marker="o", label="Scikit-learn")
plt.xlabel("Number of samples")
plt.ylabel("Time (ms)")
plt.title("Scalability Comparison")
plt.legend()
plt.grid(True)
plt.show()
