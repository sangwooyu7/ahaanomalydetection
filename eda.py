import pandas as pd

# Path to your CSV file
csv_file_path = '/Users/alexandraholikova/Downloads/supermarket.csv'

# Placeholder list to collect all the parsed data
data = []

# Read the CSV file. Assuming no header, as each row just contains product details
with open(csv_file_path, 'r') as file:
    for row_index, line in enumerate(file):
        # Remove any surrounding whitespace and split the row on commas for each product
        products = line.strip().split(',')
        # Iterate through each product
        for product in products:
            # Split the product into its details (category, time since last scan, price)
            details = product.strip().split()  # Assuming space separation of details
            if len(details) == 3:  # Make sure there are exactly three details
                category, time_since_last_scan, price = details
                data.append([row_index + 1, category, time_since_last_scan, price])

# Convert the list to a DataFrame
columns = ['Customer_Number', 'Category', 'Time_Since_Last_Scan', 'Price']
df = pd.DataFrame(data, columns=columns)

# Show the DataFrame to verify the transformation
print(df)
