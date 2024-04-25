import pandas as pd
from apyori import apriori
from scan import Scan
from receipt import Receipt
from read import read_receipts
from read import sample_receipts

def preprocess_data(receipts):
    transactions = []
    for receipt in receipts:
        transaction = [str(scan.department) for scan in receipt.scans]
        transactions.append(transaction)
    return transactions


def compute_association_rules(transactions, min_support=0.01, min_confidence=0.5):
    rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=1)
    return list(rules)  # Convert generator to list for easier manipulation later

def extract_unusual(receipts, rules):
    unusual = []
    for receipt in receipts:
        departments = set(str(scan.department) for scan in receipt.scans)
        for rule in rules:
            for stat in rule.ordered_statistics:
                antecedent = set(stat.items_base)
                consequent = set(stat.items_add)
                combo = antecedent.union(consequent)
                if combo.issubset(departments):
                    unusual.append(combo)
    return unusual

def main():
    file_path = 'supermarket.csv'
    receipts = read_receipts(file_path)  # Assuming read_receipts() function reads the receipts
    sample = sample_receipts(receipts, 0.0001, seed=90)

    # Preprocessing data
    transactions = preprocess_data(sample)

    # Computing association rules
    rules = compute_association_rules(transactions)

    # Extracting unusual combinations
    unusual_combinations = extract_unusual(receipts, rules)

    # Print unusual combinations
    print("Unusual combinations:")
    for combo in unusual_combinations:
        print(combo)

if __name__ == "__main__":
    main()
