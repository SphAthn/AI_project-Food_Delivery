"""
Sub module to handle missing values in the dataset. 
"""

import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans string whitespace, caps ratings, filters GPS/age anomalies, and imputes nulls."""
    df_clean = df.copy()

    # Counting the missing values and anomalies before cleaning
    total = len(df_clean)
    n_rating = int((df_clean["Delivery_person_Ratings"] > 5.0).sum())
    n_gps = int(((df_clean["Restaurant_latitude"] <= 0)
            | (df_clean["Restaurant_longitude"] <= 0)
            | (df_clean["Delivery_location_latitude"] <= 0)
            | (df_clean["Delivery_location_longitude"] <= 0)).sum())
    n_age = int((df_clean["Delivery_person_Age"] < 18).sum())
    n_nan = int(df_clean.isna().any(axis=1).sum())
    print("\nData before cleaning:")
    print(f"Ratings > 5.0             : {n_rating} ({n_rating / total:.2%})")
    print(f"GPS coordinates <= 0      : {n_gps} ({n_gps / total:.2%})")
    print(f"Couriers under 18         : {n_age} ({n_age / total:.2%})")
    print(f"Rows with a missing value : {n_nan} ({n_nan / total:.2%})")

    # Missing values per column (before cleaning)
    missing = df_clean.isna().sum()
    missing = missing[missing > 0]
    print("\nMissing values per column (before cleaning):")
    for col, n in missing.items():
        print(f"  {col:<28}: {n:>5} ({n / total:.2%})")


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
        & (df_clean["Delivery_person_Age"].isna() | (df_clean["Delivery_person_Age"] >= 18))
    )
    gps_invalid = (
        (df_clean["Restaurant_latitude"] <= 0)
        | (df_clean["Restaurant_longitude"] <= 0)
        | (df_clean["Delivery_location_latitude"] <= 0)
        | (df_clean["Delivery_location_longitude"] <= 0)
    )
    underage = df_clean["Delivery_person_Age"] < 18
    df_clean = df_clean[valid_mask].copy()

    print("\nRows removed:")
    print(f"  Invalid GPS     : {int(gps_invalid.sum())}")
    print(f"  Under 18        : {int(underage.sum())}")
    print(f"  Total removed   : {total - len(df_clean)} -> {len(df_clean)} rows kept")

    # 4. Impute continuous and count numerical features with medians
    imputed = {}
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        imputed[col] = (int(df_clean[col].isna().sum()), df_clean[col].median())

    df_clean["Delivery_person_Age"] = df_clean["Delivery_person_Age"].fillna(
        df_clean["Delivery_person_Age"].median()
    )
    df_clean["Delivery_person_Ratings"] = df_clean[
        "Delivery_person_Ratings"
    ].fillna(df_clean["Delivery_person_Ratings"].median())
    df_clean["multiple_deliveries"] = df_clean["multiple_deliveries"].fillna(
        df_clean["multiple_deliveries"].median()
    )
    print("\nNumeric missing values replaced by the median:")
    for col, (n, med) in imputed.items():
        print(f"  {col:<28}: {n:>5} replaced (median = {med})")

    # 5. Replace string representation of nulls in categorical columns
    categorical_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "City",
        "Festival",
        "Time_Orderd",
    ]
    for cat_col in categorical_cols:
        df_clean[cat_col] = df_clean[cat_col].replace(
            ["nan", "NaN", "", "NaN "], "Unknown"
        )
        df_clean[cat_col] = df_clean[cat_col].fillna("Unknown")
    
    print("\nMissing text values tagged as 'Unknown':")
    for cat_col in categorical_cols:
        print(f"  {cat_col:<28}: {int((df_clean[cat_col] == 'Unknown').sum()):>5}")

    # Checking which columns still contain missing values
    remaining = df_clean.isna().sum()
    remaining = remaining[remaining > 0]
    print(f"\nMissing values left after cleaning: {int(remaining.sum())}")
    if len(remaining) > 0:
        print(remaining.to_string())

    return df_clean
