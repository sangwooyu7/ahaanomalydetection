import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


csv_file_path = '/Users/alexandraholikova/Downloads/supermarket_fixed2.csv'

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
                category = int(category)
                time_since_last_scan = int(time_since_last_scan)  # Convert time to integer
                price = float(price)  # Convert price to float
                data.append([row_index + 1, category, time_since_last_scan, price])

# Convert the list to a DataFrame
columns = ['Customer_Number', 'Category', 'Time_Since_Last_Scan', 'Price']
df = pd.DataFrame(data, columns=columns)

# Adding previous category for each customer to track transitions
df['prev_category'] = df.groupby('Customer_Number')['Category'].shift()

# Filter out rows where the category hasn't changed or where Time_Since_Last_Scan is NaN
df_filtered = df[(df['Category'] != df['prev_category']) & (df['Time_Since_Last_Scan'].notna())]

# Create a pivot table to find the average time between different categories
transition_times = df_filtered.pivot_table(index='prev_category', columns='Category', values='Time_Since_Last_Scan', aggfunc='mean')

# Plotting the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(transition_times, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Average Time Between Different Departments')
plt.xlabel('To Category')
plt.ylabel('From Category')

# Making the numbers in the heatmap smaller for better readability
for text in ax.texts:
    text.set_size(8)

plt.show()