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

    # Step 1: Split dataset into train, validation, and test sets and scale without leakage
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = get_split_scaled_data()

    # Step 2: Model Training, Hyperparameter Tuning, and Evaluation Comparison
    best_model_name, best_model = run_model_comparison(X_train, X_val, y_train, y_val)

    # Step 3: Verify and Load Exported Model Artifact
    model_path = "data/best_ml_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Exported model not found at {model_path}")

    loaded_best_model = joblib.load(model_path)

    # Step 4: Make predictions on test set using the best model and export CSV contract
    predictions = run_predictions(X_test, y_test, loaded_best_model, feature_names=feature_names)

    # Prolog part (Will be integrated in next phase)

    print("\nPipeline execution completed successfully!")

