def update_centroids(points, assignments, k):
    centroids = []

    for cluster in range(k):
        cluster_points = [
            points[i]
            for i in range(len(points))
            if assignments[i] == cluster
        ]

        if cluster_points:
            x = sum(p[0] for p in cluster_points) / len(cluster_points)
            y = sum(p[1] for p in cluster_points) / len(cluster_points)
            centroids.append((x, y))
        else:
            centroids.append((0,0))

    return centroids


# Test

points = [(2,3), (5,8), (1,2), (6,9), (7,7)]
assignments = [0,1,0,1,1]

print(update_centroids(points, assignments, 2))