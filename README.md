# Depop Sale Speed Predictor

This project uses a K-Nearest Neighbors (KNN) machine learning model to predict whether a product listed on Depop will sell quickly (in less than 3 days) or slowly.

## Project Structure

- `SpeedKNN.py`: The script for training and evaluating the KNN model. It processes the data, runs a 5-fold cross-validation, and saves the trained model components.
- `Predictor.py`: An interactive command-line tool that loads the saved model components and predicts the sale speed of a new product based on user input.
- `DepopStatistics - depop_data_template (1).csv`: The dataset containing sample product listings from Depop (latest version can be shown:https://docs.google.com/spreadsheets/d/1w8PZwcodUctkTVUi4TNXesmKSGu6pgMGurNeHOno2So/edit?usp=sharing).

## Features Used for Prediction

The model uses the following features to make its predictions:

- **Price**: The listing price of the item.
- **Title Word Count**: The number of words in the product title.
- **Description Word Count**: The number of words in the product description.
- **Day of the Week**: The day of the week the item was listed.
- **Is Weekend**: A flag indicating if the item was listed on a weekend.
- **Category**: The product category (e.g., 'tops', 'dresses').
- **Size**: The size of the item.

## Setup

To run this project, you need to have Python 3 and a few libraries installed.

1.  **Install Dependencies:**
    Open your terminal and run the following command to install the necessary packages:
    ```bash
    pip3 install pandas numpy scikit-learn tabulate joblib
    ```

## How to Run

The workflow is a two-step process: first you train the model, then you can make predictions.

### Step 1: Train the Model

Navigate to the project directory and run the `SpeedKNN.py` script. This only needs to be done once, or whenever you want to retrain the model.

```bash
python3 SpeedKNN.py
```

This will train the model, print the evaluation accuracy, and create several `.joblib` files in your directory. These files contain the trained model components.

### Step 2: Make a Prediction

Once the model is trained, you can run the `Predictor.py` script to start the interactive prompt.

```bash
python3 Predictor.py
```

The script will prompt you to enter the details of a new product. After you provide the information, it will display the prediction.
