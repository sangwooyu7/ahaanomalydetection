import csv
import random
from statistics import median
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from read import read_receipts
from sort import sort_receipts_by_ratio
from sort import get_sorted_receipt_indexes
from extract import extract_features
from cluster import identify_sus_cluster
from cluster import print_cluster_characteristics
from cluster import expand_suspicious_cluster

def simulate_stealing(receipts):
    sorted_receipts = sorted(receipts, key=lambda r: r.time_cost_ratio(), reverse=True)
    time_cost_ratios = [receipt.time_cost_ratio() for receipt in receipts]
    median_ratio = median(time_cost_ratios)

    # Simulate stealing for the top 50% suspicious receipts
    num_receipts_to_manipulate = int(len(receipts) * 0.42)
    for receipt in receipts[:num_receipts_to_manipulate]:
        if receipt.time_cost_ratio() > median_ratio:
            receipt.sus = True
            simulate_stealing_receipt(receipt)

    # No need for the index_receipt_dict
    receipts_with_indexes = [(index, receipt) for index, receipt in enumerate(receipts)]
    return receipts_with_indexes

def simulate_stealing_receipt(receipt):
    # Randomly decide whether to remove entire scans or reduce prices
    remove_scans = random.random() < 0.5

    if remove_scans:
        # Randomly remove some scans
        num_scans_to_remove = max(1, int(len(receipt.scans) * 0.2))
        for _ in range(num_scans_to_remove):
            if receipt.scans:
                removed_scan = receipt.scans.pop(random.randint(0, len(receipt.scans) - 1))
                receipt.total_time -= removed_scan.time
                receipt.total_cost -= removed_scan.price
                if receipt.scans:
                    receipt.scans[0].time += removed_scan.time  # Add removed time to the next scan
    else:
        # Randomly reduce the price of some scans
        for scan in receipt.scans:
            if random.random() < 0.3:
                original_price = scan.price
                scan.price *= random.uniform(0.4, 0.9)
                receipt.total_cost -= original_price - scan.price




