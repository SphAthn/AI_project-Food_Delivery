# File: main.py
# Date: 08-10-2026
# Description: Runs the full data ingestion pipeline and exports  CSV files.
# Usage: python main.py

import os 
import joblib

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

    # Step 3: Load Winning Model Artifact
    model_path = "data/best_ml_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Exported model not found at {model_path}")

    loaded_best_model = joblib.load(model_path)

    # Step 4: Evaluate Test Set Performance Metrics via ML_predictions module
    raw_predictions = run_predictions(X_test, y_test, loaded_best_model)

    # Step 5: Directly Export Artifact CSV for Prolog Symbolic Rule Engine
    if isinstance(X_test, pd.DataFrame):
        inference_df = X_test.copy()
    else:
        inference_df = pd.DataFrame(X_test, columns=feature_names)

    inference_df["Actual_Time_taken(min)"] = y_test.values
    inference_df["Predicted_ETA_ML(min)"] = raw_predictions.round(2)

    os.makedirs("data", exist_ok=True)
    out_csv = "data/ML_Inference_Predictions.csv"
    inference_df.to_csv(out_csv, index=False)

    print(f"\n -> Successfully exported ML inference predictions to '{out_csv}'")
    print(" -> Pipeline Ready for Prolog Symbolic Rule Engine Integration!")

    # Prolog part (Will be integrated in next phase)

    print("\nPipeline execution completed successfully!")

