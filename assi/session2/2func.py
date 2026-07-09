# 2. Function Using Euclidean Distance

import math

def assign_clusters(points, centroids):
    assignments = []

    for point in points:
        distances = []

        for centroid in centroids:
            distance = math.sqrt(
                (point[0]-centroid[0])**2 +
                (point[1]-centroid[1])**2
            )
            distances.append(distance)

        assignments.append(distances.index(min(distances)))

    return assignments


# Example
points = [(2,3), (5,8), (1,2), (6,9), (7,7)]
centroids = [(2,3), (7,7)]

print(assign_clusters(points, centroids))