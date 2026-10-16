# Usage: python main.py
# Description: Master execution script for the hybrid AI system data processing and ML inference engine.

import os
import joblib
import pandas as pd

from src.data_preprocessing import get_prepared_data
from src.evaluate_models import run_model_comparison


def run_pipeline():
    """Runs the full ML subsystem pipeline from preprocessing to model selection and inference."""
    print("  HYBRID PREDICTION & INFERENCE ENGINE     ")

    # Step 1: Preprocessing & Data Splits (Leakage-Free)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = (
        get_prepared_data()
    )
    print(f" -> Training set   : {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f" -> Validation set : {X_val.shape[0]} rows")
    print(f" -> Test set       : {X_test.shape[0]} rows\n")

    # Step 2: Model Tuning, Cross-Validation & Comparison
    run_model_comparison()

    # Step 3: Load Winning Model for Inference Generation
    model_path = "data/best_ml_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Exported model not found at {model_path}")

    best_model = joblib.load(model_path)

    # Generate Baseline Predictive ETAs on Test Set
    raw_predictions = best_model.predict(X_test)

    # Create Artifact for Symbolic Rule Engine
    inference_df = pd.DataFrame(X_test, columns=feature_names)
    inference_df["Actual_Time_taken(min)"] = y_test.values
    inference_df["Predicted_ETA_ML(min)"] = raw_predictions.round(2)

    os.makedirs("data", exist_ok=True)
    out_csv = "data/ML_Inference_Predictions.csv"
    inference_df.to_csv(out_csv, index=False)

    print(f" -> Successfully exported ML inference predictions to '{out_csv}'")
    print(" -> Pipeline Ready for Prolog Symbolic Rule Engine Integration!")
    print("                  ML SUBSYSTEM EXECUTION COMPLETE               ")


if __name__ == "__main__":
    run_pipeline()