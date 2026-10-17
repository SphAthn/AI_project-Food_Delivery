
# File: inspect_model.py
# Usage: python inspect_model.py
# Description: Inspects serialized model parameters, scaler statistics, and feature importances.

import os
import joblib
import pandas as pd


def inspect_winning_model():
    model_path = "data/best_ml_model.pkl"
    predictions_path = "data/ML_Inference_Predictions.csv"

    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please run main.py first.")
        return

    # 1. Load exported model
    model = joblib.load(model_path)

    print("                WINNING MODEL SUMMARY                   ")
    print("Model Type :", type(model).__name__)
    print("Hyperparams:", model.get_params())

    # 2. Extract Feature Importances if available (Tree models like XGBoost / Random Forest)
    if hasattr(model, "feature_importances_") and os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["Actual_Time_taken(min)", "Predicted_ETA_ML(min)"]
        ]

        if len(feature_cols) == len(model.feature_importances_):
            importances = pd.Series(
                model.feature_importances_, index=feature_cols
            ).sort_values(ascending=False)

            print("            TOP 10 MOST IMPORTANT FEATURES              ")
            print(importances.head(10).round(4).to_string())
        else:
            print("\nNote: Feature count mismatch between saved dataset and model importances.")
    elif hasattr(model, "estimators_"):
        print("\nNote: Stacking Ensemble model loaded. Base estimators:", [e[0] for e in model.estimators])
    else:
        print("\nNote: Selected model type does not support direct tree feature_importances_ attribute.")


if __name__ == "__main__":
    inspect_winning_model()
