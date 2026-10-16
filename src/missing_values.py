# Usage: Imported by src.data_preprocessing
# Description: Cleans whitespace, filters invalid GPS/underage couriers, caps ratings, and imputes null values.

import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans string whitespace, caps ratings, filters GPS/age anomalies, and imputes missing values.

    Args:
        df: Raw input DataFrame.

    Returns:
        pd.DataFrame: Sanitized and imputed DataFrame.
    """
    df_clean = df.copy()

    # 1. Strip whitespace across string/object columns
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Cap Ratings > 5.0 at 5.0
    df_clean.loc[
        df_clean["Delivery_person_Ratings"] > 5.0, "Delivery_person_Ratings"
    ] = 5.0

    # 3. Filter invalid GPS coordinates (<= 0) and underage couriers (< 18)
    valid_mask = (
        (df_clean["Restaurant_latitude"] > 0)
        & (df_clean["Restaurant_longitude"] > 0)
        & (df_clean["Delivery_location_latitude"] > 0)
        & (df_clean["Delivery_location_longitude"] > 0)
        & (
            df_clean["Delivery_person_Age"].isna()
            | (df_clean["Delivery_person_Age"] >= 18)
        )
    )
    df_clean = df_clean[valid_mask].copy()

    # 4. Impute continuous and count numerical features with medians
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 5. Tag missing categorical values as 'Unknown'
    categorical_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "City",
        "Festival",
        "Time_Orderd",
    ]
    for cat_col in categorical_cols:
        if cat_col in df_clean.columns:
            df_clean[cat_col] = df_clean[cat_col].replace(
                ["nan", "NaN", "", "NaN "], "Unknown"
            )
            df_clean[cat_col] = df_clean[cat_col].fillna("Unknown")

    return df_clean