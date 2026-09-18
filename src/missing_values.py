"""
Data cleaning utilities for the Zomato ETA prediction project.

Handles: coordinate sign-fix, placeholder (0,0) coordinate removal, missing value
imputation, and duplicate removal. Operates purely on an in-memory DataFrame --
no file I/O here (that's load_dataset's job).
"""

import pandas as pd 


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Zomato dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset, as returned by load_dataset().

    Returns
    -------
    pd.DataFrame
        A cleaned copy of df: fixed coordinate sign errors, invalid (~0,0)
        placeholder coordinates removed, missing values imputed/dropped, and
        duplicate rows removed.
    """
    df = df.copy()
    n_original = df.shape[0]

    # -----------------------------------------------------------------
    # 1. Fix coordinate sign errors (India's lat/long are both positive)
    # -----------------------------------------------------------------
    coord_cols = ["Restaurant_latitude", "Restaurant_longitude",
                  "Delivery_location_latitude", "Delivery_location_longitude"]
    n_negative = (df[coord_cols] < 0).any(axis=1).sum()
    df[coord_cols] = df[coord_cols].abs()
    print(f"Fixed sign errors on {n_negative} rows with negative latitude/longitude values.")

    # -----------------------------------------------------------------
    # 2. Remove rows with placeholder (~0,0) coordinates
    #    (no real Indian restaurant/delivery location falls this close to (0,0);
    #    cannot be imputed meaningfully, so these rows are dropped)
    # -----------------------------------------------------------------
    invalid_coords = (
        (df["Restaurant_latitude"].abs() < 1) & (df["Restaurant_longitude"].abs() < 1)
    ) | (
        (df["Delivery_location_latitude"].abs() < 1) & (df["Delivery_location_longitude"].abs() < 1)
    )
    print(f"Dropping {invalid_coords.sum()} rows with placeholder (~0,0) coordinates.")
    df = df[~invalid_coords].copy()

    # -----------------------------------------------------------------
    # 3. Missing values
    # -----------------------------------------------------------------
    print("\nMissing values BEFORE cleaning:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0] if missing_before.sum() > 0 else "None found.")

    # Numeric columns -> impute with median (robust to outliers)
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        if col in df.columns:
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: filled {n_missing} missing values with median ({median_val})")

    # Categorical columns -> impute with mode (most frequent category)
    for col in ["Weather_conditions", "Road_traffic_density", "Festival", "City"]:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"  {col}: filled {n_missing} missing values with mode ('{mode_val}')")

    # Time_Orderd: a missing order time can't be safely imputed (it would fabricate a
    # specific clock time), and it affects any time-of-day features -> drop those rows.
    if "Time_Orderd" in df.columns:
        n_missing_time = df["Time_Orderd"].isnull().sum()
        df = df.dropna(subset=["Time_Orderd"]).copy()
        print(f"  Time_Orderd: dropped {n_missing_time} rows with missing order time")

    print("\nMissing values AFTER cleaning (per column):")
    print(df.isnull().sum())
    print(f"Total missing values remaining: {df.isnull().sum().sum()}")

    # -----------------------------------------------------------------
    # 4. Duplicates
    # -----------------------------------------------------------------
    n_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows found: {n_dupes}")
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"Dropped duplicates. New shape: {df.shape}")

    print(f"\nFinal cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns "
          f"({df.shape[0] / n_original:.1%} of original {n_original} rows retained)")

    return df