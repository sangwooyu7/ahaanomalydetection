import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from read import read_receipts
from extract import extract_features
from cluster import identify_sus_cluster
from cluster import print_cluster_characteristics
from cluster import expand_suspicious_cluster

# Read in file as Receipts
file_path = 'supermarket.csv'
case = read_receipts(file_path)

# Get characteristics of Receipts
features = extract_features(case)
X = np.array(features)

# Features are too many, we consider only the most distinct ones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Consider N clusters, subject to change
optimal_k = 4

num_runs = 100
sus_counts = defaultdict(int)

for _ in range(num_runs):
    kmeans = KMeans(n_clusters=optimal_k, random_state=None)
    kmeans.fit(X_pca)
    labels = kmeans.fit_predict(X_pca)

    # Assuming sus cluster is the smallest; +-15%
    sus_cluster = identify_sus_cluster(kmeans)
    labels_spot_check = expand_suspicious_cluster(kmeans, X_pca, 150)

    for i, label in enumerate(labels_spot_check):
        if label != sus_cluster:
            sus_counts[i] += 1

threshold = num_runs * 0.6  # Adjust this value as needed

suspicious_indexes = [i for i, count in sus_counts.items() if count >= threshold]
sorted_sus_counts = sorted(sus_counts.items(), key=lambda x: x[1], reverse=True)
top_sus_indexes = [i for i, count in sorted_sus_counts[:150]]
non_suspicious_indexes = [i for i in range(len(case)) if i in top_sus_indexes]


with open('sus.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for index, count in non_suspicious_indexes:
        writer.writerow([index, count])
        case[index].sus = True  # Mark the receipt as suspicious

print(f"{len(non_suspicious_indexes)} receipts marked as suspicious")

leading_index, leading_count = sorted_sus_counts[0]
last_index, last_count = sorted_sus_counts[149]

print(f"The leading suspicious index {leading_index} was classified as suspicious {leading_count} times.")
print(f"The last suspicious index {last_index} was classified as suspicious {last_count} times.")