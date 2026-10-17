"""
Feature Engineering Module
This module contains functions for feature extraction, encoding, and scaling of the dataset.
"""

import pandas as pd


def process_features(df_clean: pd.DataFrame):
    """Extracts order hour, generates one-hot encoding, and scales continuous features."""
    df_processed = df_clean.copy()

    # 1. Extract Order Hour
    # Time_Orderd has two formats: text "HH:MM" or a fraction of a day (0.8333 = 20:00)
    hours_text = pd.to_datetime(
        df_processed["Time_Orderd"], format="%H:%M", errors="coerce"
    ).dt.hour
    fraction = pd.to_numeric(df_processed["Time_Orderd"], errors="coerce")
    hours_fraction = (fraction * 24 * 60).round() // 60 % 24
    df_processed["order_hour"] = hours_text.fillna(hours_fraction)

    n_bad_hours = int(df_processed["order_hour"].isna().sum())
    print(f"Order hours that could not be read (replaced by median): {n_bad_hours}")
    bad = df_processed["order_hour"].isna()
    print(df_processed.loc[bad, "Time_Orderd"].value_counts())

    df_processed["order_hour"] = df_processed["order_hour"].fillna(
        df_processed["order_hour"].median()
    )

    # 2. Categorical One-Hot Encoding
    categorical_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_vehicle",
        "Festival",
        "City",
        "Type_of_order",
    ]
    df_encoded = pd.get_dummies( df_processed, columns=categorical_cols, drop_first=False, dtype=int)

    # Drop identifiers, raw dates/times and raw GPS (replaced by distance_km and order_hour)
    columns_to_drop = [
        "ID", "Delivery_person_ID", "Order_Date", "Time_Orderd", "Time_Order_picked",
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
    ]
    df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

    print(f"Encoding: {df_processed.shape[1]} columns -> {df_encoded.shape[1]} columns (one-hot, ID/date/GPS dropped)")
    
    return df_processed, df_encoded
