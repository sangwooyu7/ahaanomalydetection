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
from associate import extract_unusual
from associate import preprocess_data
from associate import compute_association_rules
from collections import defaultdict

def extract_features(receipts, min_confidence=0.5):
    features = []
  #  transactions = preprocess_data(receipts)
  #  rules = compute_association_rules(transactions)

    for receipt in receipts:
        # Standard time, cost, num scans
        total_time = receipt.total_time
        total_cost = receipt.total_cost
        time_cost_ratio = total_cost / total_time if total_time > 0 else 0
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

        features.append([# total_time,
                         # total_cost,
                         num_scans,
                         time_variance,
                         # max_spend_dept,
                         # max_scans_dept,
                         dept_change_proportion,
                         back_and_forth,
                         time_cost_ratio,

                         # len(unusual_combos)
                         ])

    return features



