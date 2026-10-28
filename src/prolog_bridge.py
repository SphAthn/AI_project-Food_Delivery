
# File: src/prolog_bridge.py
# Usage: python prolog_main.py
# Description: File-based batch execution bridge between Python and SWI-Prolog.

import json
import os
import subprocess
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

def run_prolog_evaluation():
    cleaned_path = "data/Zomato_Cleaned.csv"
    predictions_path = "data/ML_Inference_Predictions.csv"
    prolog_file = "prolog/dispatch_rules.pl"
    input_json_path = "data/prolog_inputs.json"
    output_json_path = "data/prolog_outputs.json"
    output_path = "data/Hybrid_Inference_Outputs.csv"

    if not os.path.exists(cleaned_path) or not os.path.exists(predictions_path):
        print("Error: Required input data contracts missing in 'data/'. Please run main.py first.")
        return

    df_cleaned = pd.read_csv(cleaned_path)
    df_preds = pd.read_csv(predictions_path)

    # Merge ground-truth human-readable features with ML predictions on index alignment
    test_size = len(df_preds)
    df_merged = df_cleaned.tail(test_size).copy().reset_index(drop=True)
    df_merged["Predicted_ETA_ML(min)"] = df_preds["Predicted_ETA_ML(min)"].values

    print(f"Executing High-Speed Prolog File-Batch Engine over {len(df_merged)} test predictions...")

    # Format all input rows into a JSON dictionary payload
    input_rows = []
    for _, row in df_merged.iterrows():
        input_rows.append([
            float(row["Predicted_ETA_ML(min)"]),
            float(row["distance_km"]),
            float(row["Delivery_person_Ratings"]),
            str(row["Weather_conditions"]),
            str(row["Road_traffic_density"]),
            str(row["Festival"]),
            float(row["multiple_deliveries"])
        ])

    payload = {"data": input_rows}

    # Write inputs to JSON contract file
    with open(input_json_path, "w") as f:
        json.dump(payload, f)

    # Execute SWI-Prolog via short command string
    query = f"use_module('{prolog_file}'), process_json_batch('{input_json_path}', '{output_json_path}'), halt."
    cmd = ["swipl", "-q", "-g", query]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                results = json.load(f)

            # Cleanup temporary interchange files
            if os.path.exists(input_json_path):
                os.remove(input_json_path)
            os.remove(output_json_path)

            hybrid_etas = [round(r[0], 2) for r in results]
            prolog_penalties = [round(r[1], 2) for r in results]
            dispatch_flags = [r[2] for r in results]

            df_merged["Prolog_Penalty_Minutes"] = prolog_penalties
            df_merged["Hybrid_Final_ETA(min)"] = hybrid_etas
            df_merged["Dispatch_Status"] = dispatch_flags

            df_merged.to_csv(output_path, index=False)
            print(f"Successfully generated Hybrid Inference Contract at '{output_path}'.")

            # Evaluate Metrics
            # Use aligned actuals and ML predictions directly from the prediction contract
            actuals = df_preds["Actual_Time_taken(min)"]
            ml_preds = df_preds["Predicted_ETA_ML(min)"]
            hybrid_preds = df_merged["Hybrid_Final_ETA(min)"]
            
            ml_rmse = mean_squared_error(actuals, ml_preds)**0.5
            ml_mae = mean_absolute_error(actuals, ml_preds)
            ml_r2 = r2_score(actuals, ml_preds)

            hybrid_rmse = mean_squared_error(actuals, hybrid_preds)**0.5
            hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
            hybrid_r2 = r2_score(actuals, hybrid_preds)

            print("      BASELINE ML VS. HYBRID AI PERFORMANCE SUMMARY     ")
            print(f"Baseline ML RMSE       : {ml_rmse:.4f} min | MAE: {ml_mae:.4f} min | R2: {ml_r2:.4f}")
            print(f"Hybrid AI System RMSE : {hybrid_rmse:.4f} min | MAE: {hybrid_mae:.4f} min | R2: {hybrid_r2:.4f}")

            ml_late = (actuals > ml_preds).mean()
            hybrid_late = (actuals > hybrid_preds).mean()
            print(f"Late rate (actual > ETA): ML {ml_late:.1%} | Hybrid {hybrid_late:.1%}")

            

    except Exception as e:
        print(f"Error during Prolog file-batch execution: {e}")


if __name__ == "__main__":
    run_prolog_evaluation()
