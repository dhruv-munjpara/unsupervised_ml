import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# Load dataset
df = pd.read_csv("assi/session3/Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create and fit DBSCAN
dbscan = DBSCAN(eps=8, min_samples=5)
labels = dbscan.fit_predict(X)

# Get unique labels
unique_labels = np.unique(labels)

# Plot
plt.figure(figsize=(8,6))

for label in unique_labels:
    if label == -1:
        color = "black"
        label_name = "Noise"
    else:
        color = plt.cm.tab10(label)
        label_name = f"Cluster {label}"

    plt.scatter(
        X.iloc[labels == label, 0],
        X.iloc[labels == label, 1],
        color=color,
        label=label_name
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering")
plt.legend()
plt.grid(True)
plt.show()