import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Example Zomato-like dataset
np.random.seed(42)

X = np.random.rand(200,10)

# PCA (Reduce to 3 Components)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Cumulative Explained Variance
cum_var = np.cumsum(pca.explained_variance_ratio_)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nCumulative Explained Variance:")
print(cum_var)

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    range(1,4),
    cum_var,
    marker='o',
    linewidth=2
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)

plt.show()