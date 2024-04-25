import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Path to your CSV file
csv_file_path = '/Users/louisimhof/Desktop/University/Year 2/Period 5/eLab 2/supermarket/supermarket.csv'

# Placeholder list to collect all the parsed data
data = []

# Read the CSV file. Assuming no header, as each row just contains product details
with open(csv_file_path, 'r') as file:
    customer_number = 1  # Start with customer number 1
    for line in file:
        # Remove any surrounding whitespace and split the row on commas for each product
        products = line.strip().split(',')
        # Iterate through each product
        for product in products:
            # Split the product into its details (category, time since last scan, price)
            details = product.strip().split()  # Assuming space separation of details
            if len(details) == 3:  # Make sure there are exactly three details
                category, time_since_last_scan, price = details
                data.append([customer_number, category, time_since_last_scan, price])
        customer_number += 1  # Increment customer number for the next row

# Convert the list to a DataFrame
columns = ['Customer_Number', 'Category', 'Time_Since_Last_Scan', 'Price']
df = pd.DataFrame(data, columns=columns)

# Convert 'Price' column to float
df['Price'] = df['Price'].astype(float)

# Generate and display each plot individually
# 1. Histogram of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True)
plt.title('Histogram of Prices')
plt.show()

# 2. Histogram of Time Since Last Scan
plt.figure(figsize=(10, 6))
sns.histplot(df['Time_Since_Last_Scan'], kde=True, color='green')
plt.title('Histogram of Time Since Last Scan')
plt.show()

# 3. Boxplot of Prices by Category
plt.figure(figsize=(12, 8))
sns.boxplot(x='Category', y='Price', data=df)
plt.title('Boxplot of Prices by Category')
plt.show()

# 4. Boxplot of Time Since Last Scan by Category
plt.figure(figsize=(12, 8))
sns.boxplot(x='Category', y='Time_Since_Last_Scan', data=df)
plt.title('Boxplot of Time Since Last Scan by Category')
plt.show()

# 5. Scatterplot of Time Since Last Scan vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Time_Since_Last_Scan', y='Price', data=df)
plt.title('Scatterplot of Time Since Last Scan vs. Price')
plt.show()

# 6. Countplot of Transactions per Category
plt.figure(figsize=(12, 8))
sns.countplot(x='Category', data=df)
plt.title('Countplot of Transactions per Category')
plt.show()

# 7. Barplot of Average Price per Category
plt.figure(figsize=(12, 8))
sns.barplot(x='Category', y='Price', data=df)
plt.title('Barplot of Average Price per Category')
plt.show()

# 8. Barplot of Average Time Since Last Scan per Category
plt.figure(figsize=(12, 8))
sns.barplot(x='Category', y='Time_Since_Last_Scan', data=df)
plt.title('Barplot of Average Time Since Last Scan per Category')
plt.show()

# 9. Heatmap of Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# 11. Histogram for the price of items
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.histplot(df['Price'].astype(float), kde=True)
plt.title('Histogram of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Histogram for time since last scan
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
sns.histplot(df['Time_Since_Last_Scan'].astype(float), kde=True, color='green')
plt.title('Histogram of Time Since Last Scan')
plt.xlabel('Time Since Last Scan')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 12. Items Bought vs. Shopping Time
# Placeholder list to collect all the parsed data
data = []

# Read the CSV file. Assuming no header, as each row just contains product details
with open(csv_file_path, 'r') as file:
    for row_index, line in enumerate(file):
        # Remove any surrounding whitespace and split the row on commas
        transactions = line.strip().split(',')
        # Initialize variables to track customer data
        customer_time = 0.0
        customer_items = 0
        # Iterate through each transaction
        for transaction in transactions:
            # Split the transaction into its details (category, time since last scan, price)
            details = transaction.strip().split()
            if len(details) == 3:  # Make sure there are exactly three details
                _, time_since_last_scan, _ = details
                # Accumulate time spent and count of items for the customer
                customer_time += float(time_since_last_scan)
                customer_items += 1
        # Append parsed data for the customer
        data.append({
            'Customer_Number': row_index + 1,
            'Shopping_Time': customer_time,
            'Items_Bought': customer_items
        })

# Create a DataFrame from the parsed data
df = pd.DataFrame(data)

# Visualize the relationship between shopping time and items bought
plt.figure(figsize=(10, 6))
plt.scatter(df['Items_Bought'], df['Shopping_Time'], alpha=0.5)
plt.title('Shopping Time vs. Items Bought')
plt.xlabel('Number of Items Bought')
plt.ylabel('Total Shopping Time')
plt.grid(True)
plt.show()

# Assuming df is your DataFrame with 'Shopping_Time' and 'Items_Bought' columns

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Shopping_Time', 'Items_Bought']])

# Choose the number of clusters (you can adjust this based on your data and needs)
num_clusters = 5

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)


# ++++++++++++++++++++++++++++++++++++++++++++++++++ #

plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Shopping_Time', data=df)
plt.title('Distribution of Shopping Time by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Shopping Time')
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Items_Bought'], cluster_data['Shopping_Time'], label=f'Cluster {cluster}', alpha=0.5)

# Plot cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='x', color='red', s=100, label='Cluster Centers')

plt.title('Shopping Time vs. Items Bought (Clustered)')
plt.xlabel('Number of Items Bought')
plt.ylabel('Total Shopping Time')
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers, annot=True, fmt='.2f', cmap='coolwarm', yticklabels=[f'Cluster {i}' for i in range(num_clusters)])
plt.title('Cluster Centers: Shopping Time vs. Items Bought')
plt.xlabel('Feature')
plt.ylabel('Cluster')
plt.show()

# 13. Line plot of price vs. time since last scan per category

# Placeholder list to collect all the parsed data
data = []

# Read the CSV file for product details
with open(csv_file_path, 'r') as file:
    for row_index, line in enumerate(file):
        # Remove any surrounding whitespace and split the row on commas for each product
        products = line.strip().split(',')
        # Iterate through each product
        for product in products:
            # Split the product into its details (category, time since last scan, price)
            details = product.strip().split()
            if len(details) == 3:  # Make sure there are exactly three details
                category, time_since_last_scan, price = details
                data.append([row_index + 1, category, time_since_last_scan, price])

# Convert the list to a DataFrame
columns = ['Customer_Number', 'Category', 'Time_Since_Last_Scan', 'Price']
df = pd.DataFrame(data, columns=columns)

# Convert 'Price' column to float
df['Price'] = df['Price'].astype(float)

# Descriptive statistics for numerical columns
print(df.describe())

# Frequency of categories
print(df['Category'].value_counts())

# Example: Time vs. Price (Overall)
plt.figure(figsize=(12, 6))
sns.lineplot(x='Time_Since_Last_Scan', y='Price', data=df, estimator=np.mean)
plt.title('Mean Price Over Time Since Last Scan')
plt.xlabel('Time Since Last Scan')
plt.ylabel('Mean Price')
plt.show()

# Example: Time vs. Price for each Category
categories = df['Category'].unique()

for category in categories:
    subset_df = df[df['Category'] == category]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time_Since_Last_Scan', y='Price', data=subset_df, estimator=np.mean)
    plt.title(f'Mean Price Over Time Since Last Scan for Category: {category}')
    plt.xlabel('Time Since Last Scan')
    plt.ylabel('Mean Price')
    plt.legend([category], loc='best')
    plt.show()

