import pandas as pd
import csv
from read import read_receipts
from read import index_receipts
from steal import simulate_stealing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_and_evaluate_model(receipts):
    # Extract features and labels from the receipts
    features = []
    labels = []
    for index, receipt in receipts:
        receipt_features = receipt.extract_features()
        features.append(receipt_features)
        labels.append(1 if receipt.sus else 0)

    # Convert features and labels to pandas DataFrame and Series
    features_df = pd.DataFrame(features)
    labels_series = pd.Series(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_series, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = rf_classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return rf_classifier


def train_and_evaluate_model_xgb(receipts):
    # Extract features and labels from the receipts
    features = []
    labels = []
    for index, receipt in receipts:
        receipt_features = receipt.extract_features()
        features.append(receipt_features)
        labels.append(1 if receipt.sus else 0)

    # Convert features and labels to pandas DataFrame and Series
    features_df = pd.DataFrame(features)
    labels_series = pd.Series(labels)

    # Create a StratifiedKFold object for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train the XGBoost Classifier with default hyperparameters
    xgb_classifier = XGBClassifier(random_state=42, use_label_encoder=False)

    # Perform cross-validation and evaluate the model
    cv_scores = []
    for train_index, test_index in cv.split(features_df, labels_series):
        X_train, X_test = features_df.iloc[train_index], features_df.iloc[test_index]
        y_train, y_test = labels_series.iloc[train_index], labels_series.iloc[test_index]

        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)


    # Train the model on the entire dataset
    xgb_classifier.fit(features_df, labels_series)

    # Evaluate the model on the entire dataset
    y_pred = xgb_classifier.predict(features_df)
    print("\nClassification Report:")
    print(classification_report(labels_series, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels_series, y_pred))

    return xgb_classifier

def get_suspicious_receipt_indexes(receipts, model, output_file):
    suspicious_indexes = []
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for index, receipt in receipts:
            receipt_features = receipt.extract_features()
            if model.predict([receipt_features])[0] == 1:  # Pass a list with a single sample
                writer.writerow([index])
                suspicious_indexes.append(index)

    return suspicious_indexes


# Read in file as Receipts
file_path = 'supermarket.csv'
supermarket = read_receipts(file_path)
receipts = simulate_stealing(supermarket)

case_path = 'case16.csv'
case = index_receipts(case_path)

# Train and evaluate the model
model_rf = train_and_evaluate_model(receipts)
model_xgb = train_and_evaluate_model_xgb(receipts)
# Get the indexes of suspicious receipts
suspicious_indexes_rf = get_suspicious_receipt_indexes(case, model_rf, 'sus_rf.csv')
suspicious_indexes_xgb = get_suspicious_receipt_indexes(case, model_xgb, 'sus_xgb.csv')