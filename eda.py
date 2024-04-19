import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
#print(df)

# Descriptive statistics for numerical columns
print(df.describe())

# Frequency of categories
print(df['Category'].value_counts())

plt.figure(figsize=(12, 6))

# Histogram for the price of items
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

#Box Plots by Category
plt.figure(figsize=(14, 7))
sns.boxplot(x='Category', y='Price', data=df)
plt.title('Box Plot of Prices by Category')
plt.xlabel('Category')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotate category labels for better readability
plt.show()

# Converting all columns to numeric for correlation calculation
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Calculating the correlation matrix
df_numeric = df[['Category', 'Time_Since_Last_Scan', 'Price']].apply(pd.to_numeric, errors='coerce')

# Calculating the correlation matrix for the numeric columns
corr = df_numeric.corr()

# Plotting the heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Excluding Customer Number')
plt.show()

#scatterplot price and time since last scan
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Time_Since_Last_Scan', y='Price', data=df_numeric)
plt.title('Scatter Plot of Price vs. Time Since Last Scan')
plt.xlabel('Time Since Last Scan')
plt.ylabel('Price')
plt.show()
