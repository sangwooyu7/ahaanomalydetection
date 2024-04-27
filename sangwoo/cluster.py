import numpy as np

# Assuming sus cluster is smallest (+- 15% because of Checker)
def identify_sus_cluster(kmeans):
    cluster_sizes = [np.sum(kmeans.labels_ == i) for i in range(kmeans.n_clusters)]
    return np.argmin(cluster_sizes)

def expand_suspicious_cluster(kmeans, X_pca, target_size=150):
    suspicious_cluster_label = identify_sus_cluster(kmeans)
    suspicious_cluster_centroid = kmeans.cluster_centers_[suspicious_cluster_label]

    distances = np.linalg.norm(X_pca - suspicious_cluster_centroid, axis=1)
    non_suspicious_indices = np.where(kmeans.labels_ != suspicious_cluster_label)[0]
    sorted_indices = non_suspicious_indices[np.argsort(distances[non_suspicious_indices])]

    current_size = np.sum(kmeans.labels_ == suspicious_cluster_label)
    receipts_to_add = min(target_size - current_size, len(sorted_indices))

    new_labels = kmeans.labels_.copy()
    new_labels[sorted_indices[:receipts_to_add]] = suspicious_cluster_label

    return new_labels

def print_cluster_characteristics(X, labels, optimal_k):
    for i in range(optimal_k):
        cluster_indices = np.where(labels == i)[0]
        cluster_features = X[cluster_indices]
        print(f"Cluster {i + 1}:")
        print("Size:", len(cluster_indices))
        print("Mean Features:", np.mean(cluster_features, axis=0))

