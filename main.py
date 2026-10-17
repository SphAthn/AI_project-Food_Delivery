# File: main.py
# Date: 08-10-2026
# Description: Runs the full data ingestion pipeline and exports  CSV files.
# Usage: python main.py

from src.data_preprocessing import get_prepared_data
from src.split_scale import get_split_scaled_data 
from src.ML_model_evaluation import run_model_comparison
from src.ML_predictions import run_predictions


# Parameters
preprocessing_needed = True  # Set to True if you want to run the preprocessing pipeline and generate CSVs


if __name__ == "__main__":
    print("Starting Pipeline Execution...")

    if preprocessing_needed:
        get_prepared_data()

    # Machine Learning Model Training and Evaluation
    # Splitting the dataset into training, validation, and test sets, and scaling continuous features without leakage.
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = get_split_scaled_data()

    # Running the model comparison and evaluation process
    best_model_name, best_model = run_model_comparison(X_train, X_val, y_train, y_val)

    # Make predictions on the test set using the best model
    predictions = run_predictions(X_test, y_test, best_model)

    # Prolog part

    print("\nPipeline execution completed successfully!")

