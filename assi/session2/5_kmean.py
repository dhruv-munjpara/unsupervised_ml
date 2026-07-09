import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset (Restaurant Rating, Price)
data = [
    [4.5,500],
    [4.7,650],
    [4.2,450],
    [3.8,300],
    [3.9,320],
    [2.5,150],
    [2.8,180],
    [4.8,700],
    [4.1,400],
    [3.0,220]
]

wcss = []

for k in range(1,7):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(data)
    wcss.append(model.inertia_)

plt.plot(range(1,7), wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()