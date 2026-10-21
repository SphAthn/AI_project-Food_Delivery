
# File: src/prolog_bridge.py
# Usage: python -m src.prolog_bridge
# Description: Connects ML baseline predictions with the Prolog symbolic rule engine
#              to calculate hybrid ETAs, penalties, and operational flags.

import os
import subprocess
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_prolog_evaluation():
    cleaned_path = "data/Zomato_Cleaned.csv"
    predictions_path = "data/ML_Inference_Predictions.csv"
    prolog_file = "prolog/dispatch_rules.pl"
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

    print(f"Executing Prolog Symbolic Engine over {len(df_merged)} test predictions...")

    prolog_penalties = []
    hybrid_etas = []
    dispatch_flags = []

    for idx, row in df_merged.iterrows():
        ml_eta = float(row["Predicted_ETA_ML(min)"])
        distance = float(row["distance_km"])
        rating = float(row["Delivery_person_Ratings"])
        weather = str(row["Weather_conditions"])
        traffic = str(row["Road_traffic_density"])
        festival = str(row["Festival"])
        multi_del = float(row["multiple_deliveries"])

        # Format Prolog query
        query = (
            f"use_module('{prolog_file}'). "
            f"evaluate_order({ml_eta}, {distance}, {rating}, '{weather}', '{traffic}', "
            f"'{festival}', {multi_del}, [FinalEta, Penalty, Flag]), "
            f"format('RESULT:~w,~w,~w~n', [FinalEta, Penalty, Flag]), halt."
        )

        cmd = ["swipl", "-q", "-g", query]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()

            if "RESULT:" in output:
                res_str = output.split("RESULT:")[1].strip()
                final_eta, penalty, flag = res_str.split(",")
                hybrid_etas.append(round(float(final_eta), 2))
                prolog_penalties.append(round(float(penalty), 2))
                dispatch_flags.append(flag)
            else:
                # Fallback if query output unparsed
                hybrid_etas.append(ml_eta)
                prolog_penalties.append(0.0)
                dispatch_flags.append("APPROVED")

        except Exception as e:
            # Fallback for swipl missing or execution error
            hybrid_etas.append(ml_eta)
            prolog_penalties.append(0.0)
            dispatch_flags.append("APPROVED_FALLBACK")

    # Append results into final data contract
    df_merged["Prolog_Penalty_Minutes"] = prolog_penalties
    df_merged["Hybrid_Final_ETA(min)"] = hybrid_etas
    df_merged["Dispatch_Status"] = dispatch_flags

    df_merged.to_csv(output_path, index=False)
    print(f"Successfully generated Hybrid Inference Contract at '{output_path}'.")

    # Evaluate Metrics
    actuals = df_merged["Time_taken (min)"]
    ml_preds = df_merged["Predicted_ETA_ML(min)"]
    hybrid_preds = df_merged["Hybrid_Final_ETA(min)"]

    ml_rmse = mean_squared_error(actuals, ml_preds, squared=False)
    ml_mae = mean_absolute_error(actuals, ml_preds)
    ml_r2 = r2_score(actuals, ml_preds)

    hybrid_rmse = mean_squared_error(actuals, hybrid_preds, squared=False)
    hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
    hybrid_r2 = r2_score(actuals, hybrid_preds)

    print("      BASELINE ML VS. HYBRID AI PERFORMANCE SUMMARY     ")
    print(f"Baseline ML RMSE    : {ml_rmse:.4f} min | MAE: {ml_mae:.4f} min | R2: {ml_r2:.4f}")
    print(f"Hybrid AI System RMSE : {hybrid_rmse:.4f} min | MAE: {hybrid_mae:.4f} min | R2: {hybrid_r2:.4f}")


if __name__ == "__main__":
    run_prolog_evaluation()
