import csv

def export(receipts, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for receipt in receipts:
            writer.writerow([receipt.index])