import numpy as np
import joblib
import datetime
from collections import Counter
from SpeedKNN import knn_predict

# This section loads the pre-trained components that were saved by SpeedKNN.py.
try:
    scaler = joblib.load('scaler.joblib')
    label_encoder_category = joblib.load('label_encoder_category.joblib')
    label_encoder_size = joblib.load('label_encoder_size.joblib')
    X_data = joblib.load('X_data.joblib')
    y_data = joblib.load('y_data.joblib')
except FileNotFoundError:
    print("[Error] Model components not found. Please run SpeedKNN.py first to train the model and save the components.")
    exit()

# This is the main execution block that runs when the script is called.
if __name__ == '__main__':
    # Loop indefinitely to allow for multiple predictions
    while True:
        print("Enter new product details to predict its sale speed.")
        print("(Type 'exit' to quit)\n")

        try:
            # Get user input for each feature
            title = input("Title: ")
            if title.lower() == 'exit':
                break

            description = input("Description: ")
            price = float(input("Price (e.g., 25.50): "))
            category = input(f"Category {label_encoder_category.classes_}: ")
            size = input(f"Size {label_encoder_size.classes_}: ")
            
            # The user's input is processed into the same format as the training data.
            title_word_count = len(title.split())
            description_word_count = len(description.split())
            
            # Assume the listing date is today for date-related features
            today = datetime.datetime.now()
            day_of_week = today.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Use the loaded encoders to transform categorical input
            category_encoded = label_encoder_category.transform([category])[0]
            size_encoded = label_encoder_size.transform([size])[0]
            
            # Assemble the final feature vector for the new product
            product_features = [
                price,
                title_word_count,
                description_word_count,
                day_of_week,
                is_weekend,
                category_encoded,
                size_encoded
            ]
            
            # The new product's features are scaled using the loaded scaler
            product_features_scaled = scaler.transform([product_features])
            
            # The KNN function is called to get the prediction
            prediction_result = knn_predict(product_features_scaled[0], X_data, y_data)
            
            # Convert the prediction (0 or 1) into a user-friendly text
            result_text = "Sells Fast (< 3 days)" if prediction_result == 1 else "Sells Slow (>= 3 days)"
            
            print(f"\n---> Prediction: {result_text}\n")
            print("-"*30 + "\n")

        # Handle errors, such as incorrect input types or non-existent categories
        except ValueError:
            print("\n[Error] Invalid input. Please ensure the price is a number.\n")
        except Exception as e:
            print(f"\n[Error] Could not make a prediction. The category or size may not exist in the original dataset. Details: {e}\n")

