import csv
from scan import Scan
from receipt import Receipt

def read_csv(file_path):
    receipts = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            receipt = Receipt()
            scans_in_row = row[0].split(',')
            for scan_info in scans_in_row:
                department, time, price = scan_info.split()
                scan = Scan(department, int(time), float(price))
                receipt.add_scan(scan)
            receipts.append(receipt)
    return receipts

