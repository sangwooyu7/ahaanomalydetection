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
from collections import defaultdict

def extract_features(receipts):
    features = []
    for receipt in receipts:
        total_time = receipt.total_time
        total_cost = receipt.total_cost
        num_scans = len(receipt.scans)
        dept_costs = defaultdict(float)
        dept_scans = defaultdict(int)
        for scan in receipt.scans:
            dept_costs[scan.department] += scan.price
            dept_scans[scan.department] += 1
        max_spend_dept = max(dept_costs.items(), key=lambda x: x[1])[0]
        max_scans_dept = max(dept_scans.items(), key=lambda x: x[1])[0]
        features.append([total_time, total_cost, num_scans, max_spend_dept, max_scans_dept])

    return features


