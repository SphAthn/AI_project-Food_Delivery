# File: inspect_model.py
# Usage: python inspect_model.py
# Description: Inspects serialized model parameters and extracts feature importances for LightGBM, XGBoost, and Random Forest.

import os
import joblib
import pandas as pd


def inspect_winning_model():
    model_path = "data/best_ml_model.pkl"
    predictions_path = "data/ML_Inference_Predictions.csv"

    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please run main.py first.")
        return

    # 1. Load exported winning model
    model = joblib.load(model_path)

    print("                 WINNING MODEL SUMMARY                   ")
    print("Model Type :", type(model).__name__)
    if hasattr(model, "get_params"):
        print("Hyperparams:", model.get_params())
    print("=========================================================\n")

    # 2. Extract Feature Importances if available (LightGBM / XGBoost / Random Forest)
    if hasattr(model, "feature_importances_") and os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["Actual_Time_taken(min)", "Predicted_ETA_ML(min)"]
        ]

        importances_array = model.feature_importances_

        if len(feature_cols) == len(importances_array):
            importances = pd.Series(
                importances_array, index=feature_cols
            ).sort_values(ascending=False)

            # Normalize LightGBM split/gain importances to percentage scale if needed
            if importances.sum() > 0:
                importances_pct = importances / importances.sum()
            else:
                importances_pct = importances

            print("            TOP 10 MOST IMPORTANT FEATURES              ")
            print(importances_pct.head(10).round(4).to_string())
        else:
            print(f"Note: Feature count mismatch ({len(feature_cols)} cols vs {len(importances_array)} importances).")
            
    elif hasattr(model, "estimators_"):
        print("Note: Stacking Ensemble loaded. Feature importances are distributed across base estimators:")
        for name, est in zip(["xgb", "lgb", "rf", "ridge", "mlp"], model.estimators_):
            print(f" - Base Estimator: {name} ({type(est).__name__})")
    else:
        print("Note: Selected model type does not expose direct feature_importances_.")


if __name__ == "__main__":
    inspect_winning_model()
