# 3. Support Euclidean and Manhattan Distance

import math

def assign_clusters(points, centroids, distance_type="euclidean"):
    assignments = []

    for point in points:
        distances = []

        for centroid in centroids:

            if distance_type == "euclidean":
                distance = math.sqrt(
                    (point[0]-centroid[0])**2 +
                    (point[1]-centroid[1])**2
                )

            elif distance_type == "manhattan":
                distance = abs(point[0]-centroid[0]) + abs(point[1]-centroid[1])

            else:
                raise ValueError("Choose 'euclidean' or 'manhattan'")

            distances.append(distance)

        assignments.append(distances.index(min(distances)))

    return assignments


# Example
points = [(2,3), (5,8), (1,2), (6,9), (7,7)]
centroids = [(2,3), (7,7)]

print("Euclidean :", assign_clusters(points, centroids, "euclidean"))
print("Manhattan :", assign_clusters(points, centroids, "manhattan"))