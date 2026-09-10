"""
Loads the raw dataset into a pandas DataFrame. 

"""

import pandas as pd


def load_dataset(csv_name):
    """
    Load the raw Zomato delivery dataset into a pandas DataFrame.

    Parameters
    ----------
    csv_name : str
        The name of the CSV file to load.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset (straight from the CSV).
    """
    return pd.read_csv(csv_name)

