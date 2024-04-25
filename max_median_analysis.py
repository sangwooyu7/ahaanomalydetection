import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np


class Receipt:
    def __init__(self, receipt_id):
        self.receipt_id = receipt_id
        self.departments = []
        self.times = []
        self.prices = []

    def get_total_price(self):
        return sum(self.prices)

    def get_total_time(self):
        return sum(self.times)

    def get_total_departments(self):
        return len(self.departments)

    def get_number_of_department_changes(self):
        """
        Calculate the number of department changes during the shopping session.
        """
        num_changes = sum(1 for i in range(1, len(self.departments)) if self.departments[i] != self.departments[i - 1])
        return num_changes

    def get_max_absence_time(self):
        """
        Calculate the maximum time spent without scanning a product.
        """
        max_absence_time = max(self.times)
        return max_absence_time


if __name__ == "__main__":
    csv_file_path = 'supermarket.csv'

    # Placeholder list to collect all the parsed data
    receipts = []

    # Read the CSV file
    with open(csv_file_path, 'r') as file:

        for row_index, line in enumerate(file):

            # Create a new Receipt object with receipt ID
            receipt = Receipt(row_index + 1)

            # Iterate through each product
            for product in line.strip().split(','):

                # Split the product into its details (category, time since last scan, price)
                details = product.strip().split()  # Assuming space separation of details

                if len(details) == 3:  # Make sure there are exactly three details
                    category, time_since_last_scan, price = details

                    # Append details to the respective lists in the Receipt object
                    receipt.departments.append(category)  # Recording department
                    receipt.times.append(int(time_since_last_scan))  # Convert to int for numeric operations
                    receipt.prices.append(float(price))  # Convert to float for numeric operations

            # Append the completed Receipt object to the list
            receipts.append(receipt)

    # Create DataFrame
    data = {
        'receipt_id': [receipt.receipt_id for receipt in receipts],
        'time_spent': [receipt.get_total_time() for receipt in receipts],
        'money_spent': [receipt.get_total_price() for receipt in receipts],
        'items_purchased': [receipt.get_total_departments() for receipt in receipts]
    }

    df = pd.DataFrame(data)

    # Add column for number of department changes
    df['department_changes'] = [receipt.get_number_of_department_changes() for receipt in receipts]
    df['max_absence_time'] = [receipt.get_max_absence_time() for receipt in receipts]

    # DataFrame ready for clustering
    print(df.head())  # Print first few rows of the DataFrame

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['time_spent', 'money_spent', 'items_purchased']])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    df['cluster'] = kmeans.labels_

    print(df.head())

    # Create 3D scatter plot
    fig = go.Figure(data=[],
                    layout=go.Layout(scene=dict(xaxis=dict(title='Time Spent'),
                                                yaxis=dict(title='Money Spent'),
                                                zaxis=dict(title='Items Purchased'))))

    # Plot each cluster
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(x=cluster_data['time_spent'],
                                   y=cluster_data['money_spent'],
                                   z=cluster_data['items_purchased'],
                                   mode='markers',
                                   marker=dict(size=5),
                                   name=f'Cluster {cluster}'))

    fig.update_layout(title='Clustering Visualization',
                      margin=dict(l=0, r=0, b=0, t=40))

    fig.show()

    cluster_medians = df.groupby('cluster').agg({'time_spent': 'median',
                                                 'department_changes': 'median',
                                                 'money_spent': 'median',
                                                 'max_absence_time': 'median'})

    cluster_medians.columns = ['median_time_spent', 'median_department_changes', 'median_money_spent',
                               'median_max_absence_time']

    print(cluster_medians)
    # Merge the DataFrames on 'cluster'
    df_merged = pd.merge(df, cluster_medians, on='cluster')

    # Calculate the variables for the new DataFrame
    df_merged['max_absence_diff'] = df_merged['max_absence_time'] - df_merged['median_max_absence_time']
    df_merged['department_changes_diff'] = df_merged['department_changes'] - df_merged['median_department_changes']
    df_merged['money_spent_diff'] = df_merged['median_money_spent'] - df_merged['money_spent']

    # Select relevant columns for the new DataFrame
    df_new = df_merged[['receipt_id', 'max_absence_diff', 'department_changes_diff', 'money_spent_diff']]

    # Print the new DataFrame
    print(df_new)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_new[['max_absence_diff', 'department_changes_diff', 'money_spent_diff']])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)

    # Add cluster labels to the DataFrame
    df_new['cluster'] = kmeans.labels_

    # Create 3D scatter plot
    fig = go.Figure(data=[],
                    layout=go.Layout(scene=dict(xaxis=dict(title='Max Absence Difference From Cluster Median'),
                                                yaxis=dict(title='Money Spent Difference From Cluster Median'),
                                                zaxis=dict(title='Department Changes Difference From Cluster Median'))))

    # Plot each cluster
    for cluster in df_new['cluster'].unique():
        cluster_data = df_new[df_new['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(x=cluster_data['max_absence_diff'],
                                   y=cluster_data['money_spent_diff'],
                                   z=cluster_data['department_changes_diff'],
                                   mode='markers',
                                   marker=dict(size=5),
                                   name=f'Cluster {cluster}'))

    fig.update_layout(title='Clustering Visualization',
                      margin=dict(l=0, r=0, b=0, t=40))

    fig.show()
