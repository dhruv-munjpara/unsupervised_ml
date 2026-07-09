import pandas as pd
from sklearn.cluster import DBSCAN

# Load dataset
df = pd.read_csv("assi/session3/Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply DBSCAN
dbscan = DBSCAN(eps=8, min_samples=5)
labels = dbscan.fit_predict(X)

# Number of clusters (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Number of noise points
noise_points = list(labels).count(-1)

print("Number of Clusters:", n_clusters)
print("Number of Noise Points:", noise_points)