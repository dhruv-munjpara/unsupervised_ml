import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load Dataset
iris = load_iris()

X = iris.data

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Centroids
centroids = kmeans.cluster_centers_

# Plot
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap='viridis',
    s=50
)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    color='red',
    marker='X',
    s=250,
    label='Centroids'
)

plt.title("KMeans Clustering on PCA-Reduced Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()