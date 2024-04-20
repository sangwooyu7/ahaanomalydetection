import csv
from receipt import Receipt
from scan import Scan  

def read_receipts(file_path):
    receipts = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            receipt = Receipt()
            for scan_data in [row.strip().split() for row in row]:
                if len(scan_data) == 3:
                    department, time, price = scan_data
                    scan = Scan(int(department), int(time), float(price))
                    receipt.add_scan(scan)
            receipts.append(receipt)
    return receipts
