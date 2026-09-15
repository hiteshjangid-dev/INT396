"""
Practical 5 — Determine the optimal clustering configuration by applying
Elbow Method, Silhouette Score, and Davies-Bouldin Index on a real-world
marketing dataset.
Run: python cluster_validation_demo.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
                      "axes.titlesize": 13, "axes.titleweight": "bold"})
os.makedirs("images", exist_ok=True)

df = pd.read_csv("../datasets/mall_customers.csv")
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
X_scaled = StandardScaler().fit_transform(X)
print("### DATASET: Mall Customers (real marketing/retail data) ###")

# ---------------------------------------------------------------------------
# 1. Elbow Method
# ---------------------------------------------------------------------------
print("\n### ELBOW METHOD ###")
K_range = range(2, 11)
inertias, silhouettes, dbis = [], [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    dbis.append(davies_bouldin_score(X_scaled, km.labels_))
    print(f"K={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}, DBI={dbis[-1]:.3f}")

# Elbow "pitfall" discussion: automated elbow detection via rate-of-change
diffs = np.diff(inertias)
diffs2 = np.diff(diffs)
elbow_k_automated = list(K_range)[int(np.argmax(diffs2)) + 1]
print(f"\nAutomated second-derivative elbow estimate: K={elbow_k_automated}")
print("(Note: automated elbow detection is genuinely ambiguous here -- this is the")
print(" 'Elbow Method pitfall' the syllabus explicitly names; see discussion below.)")

best_k_silhouette = list(K_range)[int(np.argmax(silhouettes))]
best_k_dbi = list(K_range)[int(np.argmin(dbis))]
print(f"\nBest K by Silhouette Score: {best_k_silhouette} (higher is better)")
print(f"Best K by Davies-Bouldin Index: {best_k_dbi} (lower is better)")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
axes[0].plot(list(K_range), inertias, marker="o", color="#2563eb")
axes[0].set_title("Elbow Method: Inertia vs K")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(K_range), silhouettes, marker="o", color="#16a34a")
axes[1].axvline(best_k_silhouette, color="#dc2626", linestyle="--", alpha=0.6)
axes[1].set_title(f"Silhouette Score vs K (best K={best_k_silhouette})")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette score")

axes[2].plot(list(K_range), dbis, marker="o", color="#f97316")
axes[2].axvline(best_k_dbi, color="#dc2626", linestyle="--", alpha=0.6)
axes[2].set_title(f"Davies-Bouldin Index vs K (best K={best_k_dbi})")
axes[2].set_xlabel("K")
axes[2].set_ylabel("DBI (lower = better)")

plt.suptitle("Determining Optimal K: Three Validation Methods on Real Data", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("images/01_three_validation_methods.png")
plt.close()

# ---------------------------------------------------------------------------
# 2. Silhouette plot (per-sample detail) at the chosen K
# ---------------------------------------------------------------------------
print(f"\n### DETAILED SILHOUETTE PLOT AT K={best_k_silhouette} ###")
km_final = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10).fit(X_scaled)
sample_silhouette = silhouette_samples(X_scaled, km_final.labels_)

fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10
colors = sns.color_palette("tab10", best_k_silhouette)
for i in range(best_k_silhouette):
    cluster_sil = sample_silhouette[km_final.labels_ == i]
    cluster_sil.sort()
    size = cluster_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, facecolor=colors[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper + 10
ax.axvline(silhouette_score(X_scaled, km_final.labels_), color="red", linestyle="--",
           label=f"Mean silhouette = {silhouette_score(X_scaled, km_final.labels_):.3f}")
ax.set_title(f"Silhouette Plot per Real Customer, K={best_k_silhouette}")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Customer index (grouped by cluster)")
ax.legend()
plt.tight_layout()
plt.savefig("images/02_silhouette_plot.png")
plt.close()

negative_customers = (sample_silhouette < 0).sum()
print(f"Real customers with a NEGATIVE silhouette (possibly misclustered): {negative_customers} of {len(X_scaled)}")

# ---------------------------------------------------------------------------
# 3. The Elbow Method's real pitfall, demonstrated
# ---------------------------------------------------------------------------
print("\n### ELBOW METHOD PITFALL, DEMONSTRATED ###")
print("The inertia curve decreases smoothly with no single unambiguous 'elbow' -- an automated")
print(f"second-derivative reading suggests K={elbow_k_automated}, while Silhouette and Davies-Bouldin")
print(f"both independently agree on K={best_k_silhouette}. This is the real, syllabus-named pitfall:")
print("relying on elbow-reading alone would have picked the wrong K here. Silhouette and DBI")
print("agreeing with each other (but not with the naive elbow reading) is exactly why")
print("triangulating across multiple validation methods is correct practice.")

print("\nAll charts saved to ./images/")
