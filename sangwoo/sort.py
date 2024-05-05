import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from read import read_receipts
import itertools
from collections import Counter
from export import export
from extract import extract_features
from cluster import identify_sus_cluster
from cluster import print_cluster_characteristics
from cluster import expand_suspicious_cluster


def sort_receipts_by_ratio(indexed_receipts, ratio_method, ascending=False):
    return sorted(indexed_receipts, key=lambda r: getattr(r[1], ratio_method)(), reverse=not ascending)

def get_sorted_receipt_indexes(receipts, ratio_method, top_n=300):
    # Add indices to receipts
    indexed_receipts = [(i, receipt) for i, receipt in enumerate(receipts)]

    # Sort receipts by the specified ratio method
    sorted_receipts = sort_receipts_by_ratio(indexed_receipts, ratio_method, ascending=False)[:top_n]

    # Extract indices from the sorted receipts
    sorted_indexes = [receipt[0] for receipt in sorted_receipts]

    return sorted_indexes

def find_common_receipt_indexes(receipts, top_n=300):
    # Add indices to receipts
    indexed_receipts = [(i, receipt) for i, receipt in enumerate(receipts)]

    # Sort receipts by different criteria
    time_scans_sorted = sort_receipts_by_ratio(indexed_receipts, 'time_scans_ratio', ascending=False)[:top_n]
    cost_scans_sorted = sort_receipts_by_ratio(indexed_receipts, 'cost_scans_ratio', ascending=False)[:top_n]
    time_cost_sorted = sort_receipts_by_ratio(indexed_receipts, 'time_cost_ratio', ascending=False)[:top_n]
    time_variance_sorted = sort_receipts_by_ratio(indexed_receipts, 'calculate_time_variance', ascending=False)[:top_n]

    time_scans_indexes = set(receipt[0] for receipt in time_scans_sorted)
    cost_scans_indexes = set(receipt[0] for receipt in cost_scans_sorted)
    time_cost_indexes = set(receipt[0] for receipt in time_cost_sorted)
    time_variance_indexes = set(receipt[0] for receipt in time_variance_sorted)

    # Find common indexes from any two sorted lists
    sorted_lists = [time_scans_indexes, cost_scans_indexes, time_cost_indexes, time_variance_indexes]
    common_indexes = set()
    for combo in itertools.combinations(sorted_lists, 3):
        common_indexes.update(combo[0] & combo[1] & combo[2])

    return list(common_indexes)

def export_indexes_to_csv(indexes, output_file):
    with open(output_file, 'w', newline='') as file:
        for index in indexes:
            file.write(str(index) + '\n')

# Usage
def main():
    file_path = 'case16.csv'
    receipts = read_receipts(file_path)
    # common_indexes = find_common_receipt_indexes(receipts, top_n=150)
    # export_indexes_to_csv(common_indexes, 'sus_sort.csv')
    # print(f"{len(common_indexes)} receipt indexes written to sus_sort.csv.")
    sorted_indexes = get_sorted_receipt_indexes(receipts, 'time_cost_ratio', top_n=150)
    export_indexes_to_csv(sorted_indexes, 'sus_sort.csv')
    print(f"{len(sorted_indexes)} receipt indexes written to sus_sort.csv.")


if __name__ == "__main__":
    main()