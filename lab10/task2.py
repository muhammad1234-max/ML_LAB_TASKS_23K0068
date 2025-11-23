import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel("Online Retail.xlsx")

#cleaning
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]


df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

X = df[["Quantity", "TotalAmount"]].values


X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

#k means made from scratch
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def calculate_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [np.linalg.norm(point - c) for c in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) == 0:
            new_centroids.append(X[np.random.randint(0, len(X))])
        else:
            new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    return clusters, centroids


K = 3
clusters, centroids = kmeans(X_scaled, K)

df["Cluster"] = clusters


plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200)
plt.xlabel("Quantity (scaled)")
plt.ylabel("Total Amount (scaled)")
plt.title("Customer Segmentation - K-Means (Manual Implementation)")
plt.show()


cluster_summary = df.groupby("Cluster")[["Quantity", "TotalAmount"]].mean()
print("\nCluster Summary:")
print(cluster_summary)
