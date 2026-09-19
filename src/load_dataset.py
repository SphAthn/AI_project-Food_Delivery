"""
Loads the raw dataset into a pandas DataFrame. 

"""
import os
import pandas as pd
import kagglehub
import pandas as pd


def load_dataset(dataset_path: str = "Zomato Dataset.csv") -> pd.DataFrame:
    """Ingests data via KaggleHub API with local file fallback."""
    try:
        path = kagglehub.dataset_download(
            "saurabhbadole/zomato-delivery-operations-analytics-dataset"
        )
        csv_file = os.path.join(path, "Zomato Dataset.csv")
        df = pd.read_csv(csv_file)
        print(f"Dataset successfully loaded from KaggleHub: {csv_file}")
        return df
    except Exception as e:
        print(f"KaggleHub API download failed ({e}). Attempting local fallback...")
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"Dataset successfully loaded from local fallback: {dataset_path}")
            return df
        else:
            raise FileNotFoundError(
                f"Could not load dataset from KaggleHub or local path: {dataset_path}"
            )




