"""
    This modules computes the distance between the restaurant and the delivery location for each row

    Different distances are possible:

    - Euclidean distance (straight line)
    - Manhattan distance (sum of the absolute differences of the coordinates)
    - Haversine distance (great-circle distance on a sphere, given in kilometers)

"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Euclidean distance between two points (lat1, lon1) and (lat2, lon2).
    """
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Manhattan distance between two points (lat1, lon1) and (lat2, lon2).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Haversine distance between two points (lat1, lon1) and (lat2, lon2).
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c * EARTH_RADIUS_KM


def compute_distance(df, method):
    """
    Computes the distance between the restaurant and the delivery location for each row in the dataframe and adds
    a new column 'distance' to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing restaurant and delivery location coordinates.
    method : str
        The method to use for computing the distance ('euclidean', 'manhattan', or 'haversine').

    Returns
    -------
    pd.DataFrame
        A new dataframe with an additional 'distance' column containing the computed distances.
    """

    lat1 = df["Restaurant_latitude"].values
    lon1 = df["Restaurant_longitude"].values
    lat2 = df["Delivery_location_latitude"].values
    lon2 = df["Delivery_location_longitude"].values

    if method == "euclidean":
        distance_rad = euclidean_distance(lat1, lon1, lat2, lon2)
        # Convert degrees to radians and multiply by Earth's radius
        distance_km = distance_rad * (np.pi / 180) * EARTH_RADIUS_KM  

    elif method == "manhattan":
        distance_rad = manhattan_distance(lat1, lon1, lat2, lon2)
        # Convert degrees to radians and multiply by Earth's radius
        distance_km = distance_rad * (np.pi / 180) * EARTH_RADIUS_KM  
        
    elif method == "haversine":
        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    else:
        raise ValueError("Invalid method. Choose from 'euclidean', 'manhattan', or 'haversine'.")

    df["distance"] = distance_km
    
    return df



if __name__ == "__main__":
    # Manual test 
    test_df = pd.DataFrame({"Restaurant_latitude": [22.745049], "Restaurant_longitude": [75.892471], "Delivery_location_latitude": [22.765049], "Delivery_location_longitude": [75.912471],})
    result = compute_distance(test_df, method="haversine")
    print(result)
