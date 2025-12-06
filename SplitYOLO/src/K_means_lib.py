import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


# =====================
# CONFIG
# =====================

NUM_CLUSTERS = 2
RANDOM_STATE = 42
RUNS = 10_000


# =====================
# LOAD DATA
# =====================

data = pd.read_csv("../data/data_device.csv")
raw_points = np.delete(data.values, 0, axis=1)


# =====================
# NORMALIZE
# =====================

scaler = MinMaxScaler()
points_norm = scaler.fit_transform(raw_points)


# =====================
# FEATURE WEIGHTING
# =====================

FEATURE_INDEX = {
    "Total Ram": 0,
    "Storage": 1,
    "Internet": 2,
    "Core": 3
}

WEIGHTS = {
    "Core": 1.4,
    "Total Ram": 1.2,
    "Internet": 1.1,
    "Storage": 0.65
}

points = points_norm.copy()

for feat, w in WEIGHTS.items():
    points[:, FEATURE_INDEX[feat]] *= w


# =====================
# RUN 10,000 TIMES
# =====================

total_silhouette = 0

last_labels = None
last_centroids = None

for i in range(RUNS):

    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        init="k-means++",
        n_init=20,
        random_state=RANDOM_STATE + i
    )

    labels = kmeans.fit_predict(points)
    centroids = kmeans.cluster_centers_

    total_silhouette += silhouette_score(points, labels)

    last_labels = labels
    last_centroids = centroids


# =====================
# POST PROCESS
# =====================

avg_silhouette = total_silhouette / RUNS

dist_matrix = cdist(points, last_centroids)
nearest_idx = np.argmin(dist_matrix, axis=0)

final_centers = points[nearest_idx]

clusters = []

for i in range(NUM_CLUSTERS):
    clusters.append(points[last_labels == i])


# =====================
# OUTPUT
# =====================

print("Average Silhouette:", avg_silhouette)

print("\nRepresentative centers:")
print(final_centers)

print("\nCluster sizes:")
for i, c in enumerate(clusters):
    print(f"Cluster {i}: {len(c)} samples")
