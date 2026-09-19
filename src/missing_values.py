import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans string whitespace, caps ratings, filters GPS/age anomalies, and imputes nulls."""
    df_clean = df.copy()

    # 1. Strip whitespace across object columns
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Cap Ratings > 5.0 at 5.0
    df_clean.loc[
        df_clean["Delivery_person_Ratings"] > 5.0, "Delivery_person_Ratings"
    ] = 5.0

    # 3. Filter invalid GPS coordinates and underage couriers (< 18)
    valid_mask = (
        (df_clean["Restaurant_latitude"] > 0)
        & (df_clean["Restaurant_longitude"] > 0)
        & (df_clean["Delivery_location_latitude"] > 0)
        & (df_clean["Delivery_location_longitude"] > 0)
        & (df_clean["Delivery_person_Age"] >= 18)
    )
    df_clean = df_clean[valid_mask].copy()

    # 4. Impute continuous and count numerical features with medians
    df_clean["Delivery_person_Age"] = df_clean["Delivery_person_Age"].fillna(
        df_clean["Delivery_person_Age"].median()
    )
    df_clean["Delivery_person_Ratings"] = df_clean[
        "Delivery_person_Ratings"
    ].fillna(df_clean["Delivery_person_Ratings"].median())
    df_clean["multiple_deliveries"] = df_clean["multiple_deliveries"].fillna(
        df_clean["multiple_deliveries"].median()
    )

    # 5. Replace string representation of nulls in categorical columns
    categorical_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "City",
        "Festival",
    ]
    for cat_col in categorical_cols:
        df_clean[cat_col] = df_clean[cat_col].replace(
            ["nan", "NaN", "", "NaN "], "Unknown"
        )
        df_clean[cat_col] = df_clean[cat_col].fillna("Unknown")

    return df_clean
