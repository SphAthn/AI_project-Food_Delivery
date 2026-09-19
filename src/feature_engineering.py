import pandas as pd
from sklearn.preprocessing import RobustScaler


def process_features(df_clean: pd.DataFrame):
    """Extracts order hour, generates one-hot encoding, and scales continuous features."""
    df_processed = df_clean.copy()

    # 1. Extract Order Hour
    df_processed["order_hour"] = pd.to_datetime(
        df_processed["Time_Orderd"], format="%H:%M", errors="coerce"
    ).dt.hour
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
    ]
    df_encoded = pd.get_dummies(
        df_processed, columns=categorical_cols, drop_first=False
    )

    # 3. Robust Scaling on Continuous Features ONLY
    continuous_scale_features = [
        "distance_km",
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "order_hour",
    ]
    scaler = RobustScaler()
    df_encoded[continuous_scale_features] = scaler.fit_transform(
        df_encoded[continuous_scale_features]
    )

    return df_processed, df_encoded
