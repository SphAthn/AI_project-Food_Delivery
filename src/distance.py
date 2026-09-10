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




def compute_distance(df):
    """
    Computes the distance between the restaurant and the delivery location for each row in the dataframe and adds
    a new column 'distance' to the dataframe.

    """

    lat1 = df["Restaurant_latitude"].values
    lon1 = df["Restaurant_longitude"].values
    lat2 = df["Delivery_location_latitude"].values
    lon2 = df["Delivery_location_longitude"].values

    distance_rad = euclidean_distance(lat1, lon1, lat2, lon2)
    # Convert degrees to radians and multiply by Earth's radius
    distance_km = distance_rad * (np.pi / 180) * EARTH_RADIUS_KM  
    df["distance"] = distance_km

    return df



if __name__ == "__main__":
    # Manual test 
    test_df = pd.DataFrame({"Restaurant_latitude": [22.745049], "Restaurant_longitude": [75.892471], "Delivery_location_latitude": [22.765049], "Delivery_location_longitude": [75.912471],})
    result = compute_distance(test_df)
    print(result)
