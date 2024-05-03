import csv
import random
import pandas as pd
from read import read_receipts
from receipt import Receipt
from scan import Scan


def process_and_export_receipts(filepath, output_filepath):
    # Read and process receipts
    receipts = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            receipt = Receipt()
            for scan_data in [row.strip().split() for row in row]:
                if len(scan_data) == 3:
                    department, time, price = scan_data
                    scan = Scan(int(department), int(time), float(price))
                    receipt.add_scan(scan)
            receipts.append(receipt)

    data = []
    for receipt in receipts:
        data.append({
            'Number of Scans': receipt.number_of_scans(),
            'Total Time': receipt.total_time,
            'Total Cost': receipt.total_cost,
            'Average Time per Scan': receipt.time_scans_ratio(),
            'Average Price per Scan': receipt.cost_scans_ratio(),
            'Spent the most in': receipt.biggest_spending_department()
        })
    df = pd.DataFrame(data)
    df = df.round(2)

    # Calculate additional features
    # Export to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Data successfully exported to {output_filepath}")


# Usage
def main():
    input_filepath = 'case0.csv'
    output_filepath = 'case0_straightened.csv'
    process_and_export_receipts(input_filepath, output_filepath)


if __name__ == "__main__":
    main()