import numpy as np
import pandas as pd


def compute_distance(
    df: pd.DataFrame, method: str = "haversine"
) -> pd.DataFrame:
    """Computes spatial distance between restaurant and delivery location."""
    df_out = df.copy()

    if method == "haversine":
        r = 6371.0  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(
            np.radians,
            [
                df_out["Restaurant_latitude"],
                df_out["Restaurant_longitude"],
                df_out["Delivery_location_latitude"],
                df_out["Delivery_location_longitude"],
            ],
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        df_out["distance_km"] = 2 * r * np.arcsin(np.sqrt(a))
    elif method == "manhattan":
        # Rough Manhattan approximation in km
        lat_dist = np.abs(
            df_out["Restaurant_latitude"] - df_out["Delivery_location_latitude"]
        ) * 111.0
        lon_dist = (
            np.abs(
                df_out["Restaurant_longitude"]
                - df_out["Delivery_location_longitude"]
            )
            * 111.0
            * np.cos(np.radians(df_out["Restaurant_latitude"].mean()))
        )
        df_out["distance_km"] = lat_dist + lon_dist
    else:
        raise ValueError(f"Unsupported distance method: {method}")

    return df_out
