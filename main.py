# File: main.py
# Date: 08-10-2026
# Description: Runs the full data ingestion pipeline and exports  CSV files.
# Usage: python main.py

from src.data_preprocessing import get_prepared_data
from src.split_scale import get_split_scaled_data 


# Parameters
preprocessing_needed = True  # Set to True if you want to run the preprocessing pipeline and generate CSVs


if __name__ == "__main__":
    print("Starting Pipeline Execution...")

    if preprocessing_needed:
        get_prepared_data()

    # Machine Learning Model Training and Evaluation
    # Splitting the dataset into training, validation, and test sets, and scaling continuous features without leakage.
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = get_split_scaled_data()

    

    print("\nPipeline execution completed successfully!")

