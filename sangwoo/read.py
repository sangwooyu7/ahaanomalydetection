import csv
import random
from receipt import Receipt
from scan import Scan  

def read_receipts(file_path):
    receipts = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            receipt = Receipt()  # Pass the index to the Receipt constructor
            for scan_data in [row.strip().split() for row in row]:
                if len(scan_data) == 3:
                    department, time, price = scan_data
                    scan = Scan(int(department), int(time), float(price))
                    receipt.add_scan(scan)
            receipts.append(receipt)
    num_receipts = len(receipts)
    print(f"Read in {num_receipts} receipts successfully")
    return receipts

def sample_receipts(receipts, sample_rate=0.1, seed=None):
    if seed is not None:
        random.seed(seed)
    
    sampled_receipts = []
    for receipt in receipts:
        if random.random() < sample_rate:
            sampled_receipts.append(receipt)
    return sampled_receipts