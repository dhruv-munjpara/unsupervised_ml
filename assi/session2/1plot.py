#1. Plot Points and Manually Assign Clusters

import matplotlib.pyplot as plt
import math

# Dataset
points = [(2,3), (5,8), (1,2), (6,9), (7,7)]

# Random Initial Centroids
centroid1 = (2,3)
centroid2 = (7,7)

# Plot
x = [p[0] for p in points]
y = [p[1] for p in points]

plt.scatter(x, y, color='blue', label='Points')
plt.scatter(centroid1[0], centroid1[1], color='red', marker='X', s=150, label='Centroid 1')
plt.scatter(centroid2[0], centroid2[1], color='green', marker='X', s=150, label='Centroid 2')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Food Delivery Locations")
plt.legend()
plt.grid(True)
plt.show()