import pandas as pd
from apyori import apriori
from scan import Scan
from receipt import Receipt
from read import read_receipts
from read import sample_receipts
from itertools import chain, combinations

def preprocess_data(receipts):
    transactions = []
    for receipt in receipts:
        transaction = [str(scan.department) for scan in receipt.scans]
        transactions.append(transaction)
    return transactions

def compute_association_rules(transactions, min_support=0.01, min_confidence=0.5):
    rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=1)
    return list(rules)

def generate_subsets(iterable, min_size=2):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(min_size, len(s)+1))