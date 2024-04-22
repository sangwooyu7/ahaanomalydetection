import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Path to your CSV file
csv_file_path = '/Users/louisimhof/Desktop/University/Year 2/Period 5/eLab 2/supermarket/supermarket.csv'

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

# Convert 'Price' column to float
df['Price'] = df['Price'].astype(float)

# Descriptive statistics for numerical columns
print(df.describe())

# Frequency of categories
print(df['Category'].value_counts())

# Histogram for the price of items
plt.figure(figsize=(12, 6))
sns.histplot(df['Price'].astype(float), kde=True)
plt.title('Histogram of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Histogram for time since last scan
plt.figure(figsize=(12, 6))
sns.histplot(df['Time_Since_Last_Scan'].astype(float), kde=True, color='green')
plt.title('Histogram of Time Since Last Scan')
plt.xlabel('Time Since Last Scan')
plt.ylabel('Frequency')
plt.show()

# Box Plots by Category for Price and Time Since Last Scan
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
sns.boxplot(x='Category', y='Price', data=df)
plt.title('Box Plot of Prices by Category')
plt.xlabel('Category')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotate category labels for better readability

plt.subplot(1, 2, 2)
sns.boxplot(x='Category', y='Time_Since_Last_Scan', data=df)
plt.title('Box Plot of Time Since Last Scan by Category')
plt.xlabel('Category')
plt.ylabel('Time Since Last Scan')
plt.xticks(rotation=45)  # Rotate category labels for better readability

plt.tight_layout()
plt.show()

# Correlation matrix and heatmap
plt.figure(figsize=(8, 6))
corr = df[['Time_Since_Last_Scan', 'Price']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix between Time Since Last Scan and Price')
plt.show()

# New visualization: Bar plot of mean time since last scan by category
plt.figure(figsize=(12, 8))
sns.barplot(x='Category', y='Time_Since_Last_Scan', data=df, estimator=np.mean)  # Use np.mean as estimator
plt.title('Mean Time Since Last Scan by Category')
plt.xlabel('Category')
plt.ylabel('Mean Time Since Last Scan')
plt.xticks(rotation=45)
plt.show()

# Create a figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# Scatter plot of price vs. time since last scan colored by category
sns.scatterplot(x='Time_Since_Last_Scan', y='Price', hue='Category', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Price vs. Time Since Last Scan (by Category)')
axes[0, 0].set_xlabel('Time Since Last Scan')
axes[0, 0].set_ylabel('Price')
axes[0, 0].legend(loc='upper right')

# Box plot of price distribution by category
sns.boxplot(x='Category', y='Price', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Price Distribution by Category')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Price')
axes[0, 1].tick_params(axis='x', rotation=45)

# Histogram of price distribution by category
sns.histplot(df, x='Price', hue='Category', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Price Distribution by Category')
axes[1, 0].set_xlabel('Price')
axes[1, 0].set_ylabel('Frequency')

# Histogram of time distribution by category
sns.histplot(df, x='Time_Since_Last_Scan', hue='Category', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Time Since Last Scan Distribution by Category')
axes[1, 1].set_xlabel('Time Since Last Scan')
axes[1, 1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# Descriptive statistics for numerical columns
print(df.describe())
