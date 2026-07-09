import pandas as pd
from sklearn.cluster import DBSCAN

# Load Dataset
df = pd.read_csv("assi/session3/Mall_Customers.csv")

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Different Parameters
parameters = [
    (5, 5),
    (8, 5),
    (10, 3)
]

for eps, min_samples in parameters:

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = list(labels).count(-1)

    print(f"eps = {eps}, min_samples = {min_samples}")
    print("Clusters:", clusters)
    print("Noise Points:", noise)
    print("-" * 40)

print("\nSummary:")
print("1. Smaller eps creates more clusters and more noise.")
print("2. Larger eps merges clusters and reduces noise.")
print("3. Larger min_samples increases the number of noise points.")
print("4. Proper parameter selection is important for meaningful clustering.")