
# Description: Computes spatial distance (Manhattan, Euclidean, Haversine) between restaurant and delivery locations.

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
KM_PER_DEGREE = EARTH_RADIUS_KM * np.pi / 180.0


def _degrees_to_km(lat1, lon1, lat2, lon2):
    """Converts coordinate differences in degrees to kilometers."""
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    north_south_km = np.abs(lat2 - lat1) * KM_PER_DEGREE
    east_west_km = np.abs(lon2 - lon1) * KM_PER_DEGREE * np.cos(mean_lat_rad)
    return north_south_km, east_west_km


def euclidean_distance(lat1, lon1, lat2, lon2):
    """Computes Euclidean distance between two coordinate points."""
    d1, d2 = _degrees_to_km(lat1, lon1, lat2, lon2)
    return np.sqrt(d1**2 + d2**2)


def manhattan_distance(lat1, lon1, lat2, lon2):
    """Computes Manhattan distance between two coordinate points."""
    d1, d2 = _degrees_to_km(lat1, lon1, lat2, lon2)
    return d1 + d2


def haversine_distance(lat1, lon1, lat2, lon2):
    """Computes Haversine distance on a sphere between two coordinate points."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c * EARTH_RADIUS_KM


def compute_distance(df: pd.DataFrame, method: str = "manhattan") -> pd.DataFrame:
    """Computes distance between restaurant and delivery location and adds 'distance_km' column.

    Args:
        df: DataFrame containing coordinate columns.
        method: 'euclidean', 'manhattan', or 'haversine'.

    Returns:
        pd.DataFrame with added 'distance_km' column.
    """
    df_out = df.copy()
    lat1 = df_out["Restaurant_latitude"].to_numpy()
    lon1 = df_out["Restaurant_longitude"].to_numpy()
    lat2 = df_out["Delivery_location_latitude"].to_numpy()
    lon2 = df_out["Delivery_location_longitude"].to_numpy()

    if method == "euclidean":
        df_out["distance_km"] = euclidean_distance(lat1, lon1, lat2, lon2)
    elif method == "manhattan":
        df_out["distance_km"] = manhattan_distance(lat1, lon1, lat2, lon2)
    elif method == "haversine":
        df_out["distance_km"] = haversine_distance(lat1, lon1, lat2, lon2)
    else:
        raise ValueError(
            "Invalid method. Choose from 'euclidean', 'manhattan', or 'haversine'."
        )

    return df_out