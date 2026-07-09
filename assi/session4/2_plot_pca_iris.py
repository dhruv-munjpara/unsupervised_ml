import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8,6))

colors = ['red', 'green', 'blue']
species = iris.target_names

for i in range(3):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        color=colors[i],
        label=species[i]
    )

plt.title("PCA Projection of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()