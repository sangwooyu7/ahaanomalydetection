import csv
import random
import pandas as pd
from read import read_receipts
from receipt import Receipt
from scan import Scan
import matplotlib.pyplot as plt


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
        for scan in receipt.scans:
            data.append({
                'Department': scan.department,
                'Time': scan.time,
                'Price': scan.price
            })

        # Convert list to DataFrame
    df = pd.DataFrame(data)
    df = df.round(2)

    # Calculate additional features
    # Export to CSV
    df.to_csv(output_filepath, index=False)
    department_data = df[df['Department'] == 1]
    plt.figure(figsize=(10, 6))
    plt.hist(department_data['Price'], bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Prices in Department 14')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    print(f"Data successfully exported to {output_filepath}")


# Usage
def main():
    input_filepath = 'case0.csv'
    output_filepath = 'case0_straightened_scan.csv'
    process_and_export_receipts(input_filepath, output_filepath)


if __name__ == "__main__":
    main()