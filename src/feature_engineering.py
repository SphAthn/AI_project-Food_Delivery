# Usage: Imported by src.data_preprocessing
# Description: Feature engineering module that extracts order hour and applies One-Hot Encoding without scaling.

import pandas as pd


def process_features(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts order hour, generates one-hot encoding, and drops raw metadata.

    Continuous scaling is intentionally omitted here to prevent data leakage prior
    to the Train/Validation/Test split.

    Args:
        df_clean: Sanitized and clean DataFrame containing calculated distance_km.

    Returns:
        A tuple of (unscaled_clean_df, unscaled_encoded_ml_df).
    """
    df_processed = df_clean.copy()

    # 1. Extract Order Hour
    hours_text = pd.to_datetime(
        df_processed["Time_Orderd"], format="%H:%M", errors="coerce"
    ).dt.hour
    fraction = pd.to_numeric(df_processed["Time_Orderd"], errors="coerce")
    hours_fraction = (fraction * 24 * 60).round() // 60 % 24
    df_processed["order_hour"] = hours_text.fillna(hours_fraction)

    # Impute missing order hours with median
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
    df_encoded = pd.get_dummies(
        df_processed, columns=categorical_cols, drop_first=False, dtype=int
    )

    # Drop identifiers, raw dates/times, and raw GPS coordinates
    columns_to_drop = [
        "ID",
        "Delivery_person_ID",
        "Order_Date",
        "Time_Orderd",
        "Time_Order_picked",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

    return df_processed, df_encoded