from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Dataset
iris = load_iris()

X = iris.data

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)

labels = kmeans.fit_predict(X_pca)

# Silhouette Score
score = silhouette_score(X_pca, labels)

print("Silhouette Score:", score)

print("\nInterpretation:")
print("A silhouette score close to 1 indicates well-separated clusters.")
print("A score around 0 indicates overlapping clusters.")
print("A negative score suggests poor clustering.")