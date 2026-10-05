# Student Name: Sophie Anthoine
# Student FAN: anth0095
# File: distance.py
# Date: 08-10-2026
# Description: Computes the distance in km for each order.

"""
    Different distances are possible:

    - Euclidean distance (straight line)
    - Manhattan distance (sum of the absolute differences of the coordinates)
    - Haversine distance (great-circle distance on a sphere, given in kilometers)

"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
KM_PER_DEGREE = EARTH_RADIUS_KM * np.pi / 180.0 # Length of one degree of latitude 


def _degrees_to_km(lat1, lon1, lat2, lon2):
    """
    Converts a coordinate difference in degrees into kilometres.

    One degree of latitude is always the same length but one degree of longitude is
    shorter when we are far from the equator (cos(latitude))
    """
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    north_south_km = np.abs(lat2 - lat1) * KM_PER_DEGREE
    east_west_km = np.abs(lon2 - lon1) * KM_PER_DEGREE * np.cos(mean_lat_rad)
    return north_south_km, east_west_km


def euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Euclidean distance between two points (lat1, lon1) and (lat2, lon2).
    """
    d1, d2 = _degrees_to_km(lat1, lon1, lat2, lon2)
    return np.sqrt(d1 ** 2 + d2 ** 2)

def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Manhattan distance between two points (lat1, lon1) and (lat2, lon2).
    """
    d1, d2 = _degrees_to_km(lat1, lon1, lat2, lon2)
    return d1 + d2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Haversine distance between two points (lat1, lon1) and (lat2, lon2).
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c * EARTH_RADIUS_KM


def compute_distance(df, method="manhattan"):
    """
    Computes the distance between the restaurant and the delivery location for each row in the dataframe and adds
    a new column 'distance_km' to the dataframe.

    Args:
        df: Dataframe with the four coordinate columns in COORDINATE_COLUMNS.
        method: 'euclidean', 'manhattan' or 'haversine'.

    Returns:
        A dataframe with an extra 'distance_km' column.
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
        raise ValueError("Invalid method. Choose from 'euclidean', 'manhattan', or 'haversine'.")
    return df_out



if __name__ == "__main__":
    # Manual test 
    test_df = pd.DataFrame({"Restaurant_latitude": [22.745049], "Restaurant_longitude": [75.892471], "Delivery_location_latitude": [22.765049], "Delivery_location_longitude": [75.912471],})
    result = compute_distance(test_df, method="haversine")
    print(result)
