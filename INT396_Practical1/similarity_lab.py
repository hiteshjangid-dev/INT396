import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mall_customers.csv")

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].to_numpy(float)

print("Dataset:", df.shape)
print("Missing values:")
print(df[features].isna().sum())

# Customer 1 vs Customer 2
A, B = X[0], X[1]

euclidean = np.linalg.norm(A - B)
manhattan = np.abs(A - B).sum()

cosine_distance = cosine_distances(
    A.reshape(1, -1),
    B.reshape(1, -1)
)[0, 0]

cosine_similarity_value = 1 - cosine_distance

print("\nCustomer 1:", A)
print("Customer 2:", B)
print(f"Euclidean: {euclidean:.4f}")
print(f"Manhattan: {manhattan:.4f}")
print(f"Cosine similarity: {cosine_similarity_value:.4f}")

# Top 5 nearest customers using all 200 rows
for name, metric in [("Euclidean", "euclidean"), ("Manhattan", "manhattan")]:
    distances = pairwise_distances(
        X, X[0:1], metric=metric
    ).ravel()

    order = np.argsort(distances)
    order = order[order != 0][:5]

    print(f"\n{name} nearest customers:")
    for i in order:
        print(int(df.iloc[i]["CustomerID"]), f"{distances[i]:.4f}")

cosine_distances_all = cosine_distances(X, X[0:1]).ravel()
order = np.argsort(cosine_distances_all)
order = order[order != 0][:5]

print("\nCosine nearest customers:")
for i in order:
    print(
        int(df.iloc[i]["CustomerID"]),
        f"{1 - cosine_distances_all[i]:.4f}"
    )

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

euclidean_scaled = np.linalg.norm(X_scaled[0] - X_scaled[1])
manhattan_scaled = np.abs(X_scaled[0] - X_scaled[1]).sum()
cosine_scaled = cosine_similarity(
    X_scaled[0].reshape(1, -1),
    X_scaled[1].reshape(1, -1)
)[0, 0]

print("\nAfter standardization:")
print(f"Euclidean: {euclidean_scaled:.4f}")
print(f"Manhattan: {manhattan_scaled:.4f}")
print(f"Cosine similarity: {cosine_scaled:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    s=28,
    alpha=0.7
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customers")
plt.tight_layout()
plt.show()
