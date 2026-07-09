import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans

# Load Dataset
df = pd.read_csv("assi/session3/Mall_Customers.csv")

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------- DBSCAN ----------------
dbscan = DBSCAN(eps=8, min_samples=5)
db_labels = dbscan.fit_predict(X)

# Number of clusters (excluding noise)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

# If DBSCAN finds 0 clusters, use 2
if n_clusters <= 0:
    n_clusters = 2

# ---------------- KMeans ----------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
km_labels = kmeans.fit_predict(X)

# ---------------- Plot ----------------
plt.figure(figsize=(12,5))

# DBSCAN Plot
plt.subplot(1,2,1)
plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=db_labels,
    cmap="tab10",
    s=50
)
plt.title("DBSCAN")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

# KMeans Plot
plt.subplot(1,2,2)
plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=km_labels,
    cmap="tab10",
    s=50
)
plt.title("K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()

# ---------------- Comparison ----------------
print("Comparison")
print("-" * 40)
print("DBSCAN:")
print("• Detects noise (outliers).")
print("• Works with arbitrary-shaped clusters.")
print("• Does not require specifying k.")

print("\nK-Means:")
print("• Every point belongs to a cluster.")
print("• Best for spherical clusters.")
print("• Requires the number of clusters (k).")

print("\nConclusion:")
print("DBSCAN is better for datasets with outliers and irregular cluster shapes.")
print("K-Means is better for compact, well-separated clusters.")