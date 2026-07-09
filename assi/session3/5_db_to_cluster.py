import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Create Dataset
X, y = make_moons(
    n_samples=300,
    noise=0.05,
    random_state=42
)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Number of noise points
noise_points = np.sum(labels == -1)

print("Number of Clusters:", n_clusters)
print("Number of Noise Points:", noise_points)

# Print indices of noise points
noise_indices = np.where(labels == -1)[0]

print("\nIndices of Noise Points:")
print(noise_indices)

# Plot Clusters
plt.figure(figsize=(8,6))

unique_labels = np.unique(labels)

for label in unique_labels:

    if label == -1:
        color = "black"
        label_name = "Noise"
    else:
        color = plt.cm.tab10(label)
        label_name = f"Cluster {label}"

    plt.scatter(
        X[labels == label, 0],
        X[labels == label, 1],
        color=color,
        label=label_name,
        s=40
    )

plt.title("DBSCAN Clustering on make_moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()