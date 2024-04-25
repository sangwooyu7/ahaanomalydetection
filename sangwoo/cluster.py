import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scan import Scan
from receipt import Receipt
from read import read_receipts
from read import sample_receipts
from extract import extract_features
from collections import defaultdict

file_path = 'supermarket.csv'
receipts = read_receipts(file_path)
set_seed = 999
sample = sample_receipts(receipts, 0.05, set_seed)
features = extract_features(sample)
X = np.array(features)

pca = PCA(n_components=3)  # Choose the number of components
X_pca = pca.fit_transform(X)
components = pca.components_
original_features = [# 'total_time',
                     # 'total_cost',
                     'num_scans',
                     'time_variance',
                     'dept_change_proportion',
                     'back_and_forth',
                     'time_cost_ratio']

for i, component in enumerate(components):
    print(f"Component {i+1}:")
    for feature, loading in zip(original_features, component):
        print(f"{feature}: {loading}")

inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

plt.plot(range(2, 11), inertias)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

optimal_k = 5

# Silhouette analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, labels))

plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

kkmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Analyze cluster characteristics
for i in range(optimal_k):
    cluster_indices = np.where(labels == i)[0]
    cluster_features = X[cluster_indices]
    print(f"Cluster {i + 1}:")
    print("Size:", len(cluster_indices))
    print("Mean Features:", np.mean(cluster_features, axis=0))

