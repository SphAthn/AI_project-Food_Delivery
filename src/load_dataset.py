# Usage: Imported by src.data_preprocessing
# Description: Ingests the Zomato delivery dataset via KaggleHub API with local fallback support.

import os
import kagglehub
import pandas as pd


def load_dataset(dataset_path: str = "data/Zomato Dataset.csv") -> pd.DataFrame:
    """Ingests data via KaggleHub API with local CSV file fallback.

    Args:
        dataset_path: Local relative path to the fallback CSV file.

    Returns:
        pd.DataFrame: Raw loaded dataset.
    """
    try:
        path = kagglehub.dataset_download(
            "saurabhbadole/zomato-delivery-operations-analytics-dataset"
        )
        csv_file = os.path.join(path, "Zomato Dataset.csv")
        df = pd.read_csv(csv_file)
        print(f"Dataset successfully loaded from KaggleHub API: {csv_file}")
        return df
    except Exception as e:
        print(f"KaggleHub API download failed ({e}). Attempting local fallback...")
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"Dataset successfully loaded from local fallback: {dataset_path}")
            return df
        else:
            raise FileNotFoundError(
                f"Could not load dataset from KaggleHub API or local path: {dataset_path}"
            )


if __name__ == "__main__":
    # Quick sanity check
    df_sample = load_dataset()
    print(f"Loaded shape: {df_sample.shape}")