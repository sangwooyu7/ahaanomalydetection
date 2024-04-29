import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from read import read_receipts
from extract import extract_features
from cluster import identify_sus_cluster
from cluster import print_cluster_characteristics
from cluster import expand_suspicious_cluster

# Read in file as Receipts
file_path = 'case2.csv'
case = read_receipts(file_path)

# Get characteristics of Receipts
features = extract_features(case)
X = np.array(features)

# Features are too many, we consider only the most distinct ones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Consider N clusters, subject to change
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=10)
kmeans.fit(X_pca)
labels = kmeans.fit_predict(X_pca)

# Assuming sus cluster is the smallest; +-15%
sus_cluster = identify_sus_cluster(kmeans)
labels_spot_check = expand_suspicious_cluster(kmeans, X_pca, 150)
print_cluster_characteristics(X_pca, labels_spot_check, optimal_k)

suspicious_indexes = [i for i, label in enumerate(labels_spot_check) if label == sus_cluster]
non_suspicious_indexes = [i for i, label in enumerate(labels_spot_check) if label != sus_cluster]

with open('sus.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for index in non_suspicious_indexes:
        writer.writerow([index])
        case[index].sus = True  # Mark the receipt as suspicious

print(f"{len(suspicious_indexes)} receipts marked as suspicious")