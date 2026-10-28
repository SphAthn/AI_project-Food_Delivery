# File: src/prolog_explain.py
# Usage: python explain_main.py [row_index]
# Description: Explanation of an order predicted time. (Using the same rules as the batch engine (prolog/dispatch_rules.pl).)

import subprocess
import pandas as pd

EXPLAIN_FILE = "prolog/explain_rules.pl"
HYBRID_PATH = "data/Hybrid_Inference_Outputs.csv"
PREDICTIONS_PATH = "data/ML_Inference_Predictions.csv"


def _atom(value):
    """Prolog atom helper (Stormy -> 'Stormy')."""
    text = str(value).strip().replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def explain_prediction(ml_eta, distance_km, rating, weather, traffic, festival, multiple_deliveries, verbose=True):
    """Explains a prediction. 
    Returns the explanation text (and prints it if verbose)."""
    query = (
        f"use_module('{EXPLAIN_FILE}'), "
        f"explain_order({float(ml_eta)}, {float(distance_km)}, {float(rating)}, "
        f"{_atom(weather)}, {_atom(traffic)}, {_atom(festival)}, "
        f"{float(multiple_deliveries)}), halt."
    )
    try:
        result = subprocess.run(["swipl", "-q", "-g", query], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during Prolog explanation: {e.stderr or e}")
        return None

    if verbose:
        print(result.stdout)
    return result.stdout



def explain_row(index, verbose=True , show_actual=False):
    """Explains one order of the test set (row index in Hybrid_Inference_Outputs.csv)."""
    df = pd.read_csv(HYBRID_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)
    row = df.iloc[index]

    text = explain_prediction(
        ml_eta=row["Predicted_ETA_ML(min)"],
        distance_km=row["distance_km"],
        rating=row["Delivery_person_Ratings"],
        weather=row["Weather_conditions"],
        traffic=row["Road_traffic_density"],
        festival=row["Festival"],
        multiple_deliveries=row["multiple_deliveries"],
        verbose=verbose,
    )
    if verbose and show_actual:
        actual = preds["Actual_Time_taken(min)"].iloc[index]
        print(f"Actual delivery time: {actual:.1f} min")
    return text