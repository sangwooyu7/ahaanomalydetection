import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scan import Scan
from receipt import Receipt
from read import read_receipts
from read import sample_receipts
from associate import preprocess_data
from associate import compute_association_rules
from associate import generate_subsets
from collections import defaultdict


def extract_features(receipts, min_confidence=0.5):
    features = []
    # transactions = preprocess_data(receipts)
    # rules = compute_association_rules(transactions)
    # department_subsets = defaultdict(int)

    # Mine association rules first
    # for receipt in receipts:
    #    departments = [scan.department for scan in receipt.scans]
    #    for subset in generate_subsets(departments, 2):
    #        department_subsets[frozenset(subset)] += 1
    #
    #    sorted_subsets = sorted(department_subsets.items(), key=lambda x: x[1], reverse=True)

    # top_subsets = [set(subset) for subset, _ in sorted_subsets[:10]]

    for receipt in receipts:
        # Standard time, cost, num scans
        total_time = math.log(receipt.total_time + 1)
        total_cost = math.log(receipt.total_cost + 1)
        time_cost_ratio = total_time / total_cost if total_time > 0 else 0
        time_variance = receipt.calculate_time_variance()
        num_scans = len(receipt.scans)
        dept_costs = defaultdict(float)
        dept_scans = defaultdict(int)

        # Features related to dept changes 
        dept_changes = 0
        back_and_forth = 0
        visited_depts = set()
        prev_dept = None

        for scan in receipt.scans:
            dept_costs[scan.department] += scan.price
            dept_scans[scan.department] += 1
            if prev_dept is not None and scan.department != prev_dept:
                dept_changes += 1
                if scan.department in visited_depts:
                    back_and_forth += 1
                visited_depts.add(prev_dept)
            prev_dept = scan.department
        #    unusual_combos = extract_unusual([receipt], rules)
        # Currently reduced customer spending to dept where max spending/scans happen
        # max_spend_dept = max(dept_costs.items(), key=lambda x: x[1])[0]
        # max_scans_dept = max(dept_scans.items(), key=lambda x: x[1])[0]

        dept_change_proportion = dept_changes / (num_scans - 1) if num_scans > 1 else 0
        # departments = set(scan.department for scan in receipt.scans)
        # in_top_subsets = any(departments.issuperset(subset) for subset in top_subsets)

        # departments = set(str(scan.department) for scan in receipt.scans)
        # unusual_combo = False
        # for rule in rules:
        #    antecedent = set(str(item) for item in rule.ordered_statistics[0].items_base)
        #     consequent = set(str(item) for item in rule.ordered_statistics[0].items_add)
        #    combo = antecedent.union(consequent)Q
        #    if combo.issubset(departments) and len(combo) > 1:
        #        unusual_combo = True
        #        break

        features.append([  # total_time,
            # total_cost,
            # num_scans,
            time_variance,
            # max_spend_dept,
            # max_scans_dept,
            dept_change_proportion,
            back_and_forth,
            time_cost_ratio,
            # int(in_top_subsets)
            # len(unusual_combos)
        ])

    feature_matrix = np.array(features)
    # if feature_matrix.size > 0:
    #     min_values = np.min(feature_matrix, axis=0)
    #     max_values = np.max(feature_matrix, axis=0)
    #     normalized_features = (feature_matrix - min_values) / (max_values - min_values)
    # else:
    #    normalized_features = feature_matrix
    scaler = RobustScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    return normalized_features
