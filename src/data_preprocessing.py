# Usage: Imported by main.py and model training modules
# Description: Pipeline orchestrator performing data ingestion, distance computation, split, and robust scaling.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.distance import compute_distance
from src.feature_engineering import process_features
from src.load_dataset import load_dataset
from src.missing_values import clean_dataset


def get_prepared_data(
    raw_csv_path: str = "data/Zomato Dataset.csv",
    target_col: str = "Time_taken (min)",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """Loads raw dataset, cleans, engineers features, splits data, and scales continuous attributes without leakage.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    # 1. Load, Clean, and Compute Distance
    raw_df = load_dataset(raw_csv_path)
    clean_df = clean_dataset(raw_df)
    df_dist = compute_distance(clean_df, method="manhattan")

    # 2. Extract Features & One-Hot Encode (Unscaled)
    df_processed, df_encoded = process_features(df_dist)

    # 3. Separate Features and Target Variable
    if target_col not in df_encoded.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # 4. Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    # First split off Test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split off Validation set (15% of total = 15/85 of temp)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    # 5. Fit RobustScaler ONLY on Training Set to Prevent Data Leakage
    continuous_features = [
        "distance_km",
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "order_hour",
    ]

    scaler = RobustScaler()
    
    # Copy to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_val[continuous_features] = scaler.transform(X_val[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, X.columns.tolist()
