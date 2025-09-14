import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate
import joblib

# --- KNN Algorithm Functions ---

def euclidean_distance(x1, x2):
    """Calculates the euclidean distance between two numpy arrays."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(test_sample, train_X, train_y, k=5):
    """Makes a prediction for a single test sample using the KNN algorithm."""
    distances = []

    for i in range(len(train_X)):
        dist = euclidean_distance(test_sample, train_X[i])
        distances.append((dist, train_y[i]))

    distances.sort(key=lambda x: x[0])
    k_labels = [label for (_, label) in distances[:k]]
    prediction = Counter(k_labels).most_common(1)[0][0]
    return prediction

# --- Model Evaluation Function ---

def evaluate_knn(train_X, train_y, test_X, test_y, k=5):
    """Evaluates the KNN model on a test set and returns the accuracy."""
    correct = 0
    for i in range(len(test_X)):
        pred = knn_predict(test_X[i], train_X, train_y, k)
        if pred == test_y[i]:
            correct += 1
    return correct / len(test_y)

# --- Main Execution Block ---
# The code inside this block will only run when the script is executed directly.
# It will not run when the script is imported by another file.
if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('DepopStatistics - depop_data_template (1).csv')

    # --- 1. Preprocessing and Feature Engineering ---
    df['date_listed'] = pd.to_datetime(df['date_listed'])
    df['date_sold'] = pd.to_datetime(df['date_sold'])
    df['days_to_sell'] = (df['date_sold'] - df['date_listed']).dt.days
    df['label'] = (df['days_to_sell'] < 3).astype(int)
    df['title_word_count'] = df['title'].str.split().str.len()
    df['description_word_count'] = df['description'].str.split().str.len()
    df['day_of_week'] = df['date_listed'].dt.dayofweek
    df['is_weekend'] = (df['date_listed'].dt.dayofweek >= 5).astype(int)

    label_encoder_category = LabelEncoder()
    df['category_encoded'] = label_encoder_category.fit_transform(df['category'])
    label_encoder_size = LabelEncoder()
    df['size_encoded'] = label_encoder_size.fit_transform(df['size'])

    features = [
        'price', 'title_word_count', 'description_word_count',
        'day_of_week', 'is_weekend', 'category_encoded', 'size_encoded'
    ]
    X = df[features].values
    y = df['label'].values

    # Scale the features (important for distance-based algorithms like KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2. K-Fold Cross-Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc = evaluate_knn(X_train, y_train, X_test, y_test, k=5)
        accuracies.append(acc)

    # --- 3. Display Results ---
    print("--- Model Evaluation ---")
    table_data = [[f"Experiment {i+1}", f"{acc * 100:.2f}%"] for i, acc in enumerate(accuracies)]
    table_data.append(["Average", f"{np.mean(accuracies) * 100:.2f}%"])
    headers = ["Experiment", "Accuracy"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # --- 4. Save Components for Predictor Script ---
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(label_encoder_category, 'label_encoder_category.joblib')
    joblib.dump(label_encoder_size, 'label_encoder_size.joblib')
    joblib.dump(X_scaled, 'X_data.joblib') # Save the scaled data
    joblib.dump(y, 'y_data.joblib')

    print("\nModel components saved successfully.")

