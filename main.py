"""
Main entry point for the project. 

Run with:
    python main.py

"""

import os
import numpy as np
import pandas as pd

from src.distance import compute_distance #choose 1 which one is work delete 1
#from src.distance2 import compute_distance #choose 2 choose 1 which one is work delete 1
from src.feature_engineering import process_features # Temporal Extraction & RobustScaler
from src.load_dataset import load_dataset # KaggleHub + Local Fallback Ingestion
from src.missing_values import clean_dataset # Sanitization, Filtering, and Imputation


# Configuration and Constants
RAW_CSV_NAME = "data/Zomato Dataset.csv"
CLEANED_CSV_OUT = "data/Zomato_Cleaned.csv" # Exported artifact for Symbolic Rules
ENCODED_ML_CSV_OUT = "data/Zomato_Encoded_ML.csv" # Exported artifact for ML Training


if __name__ == "__main__":

    # 1. Load the Raw Dataset
    raw_df = load_dataset(RAW_CSV_NAME)
    print(f"Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns\n")
    print("First rows of the dataset:")
    print(raw_df.head())
    
    # 2. Clean Dataset & Perform Null/Anomaly Audit
    clean_df = clean_dataset(raw_df)

    # 3. Compute Vectorized Haversine Distance (km)
    df_with_distance = compute_distance(clean_df, method="haversine") #or manhattan change "manhattan" to haversine

    print(
        f"Dataset after cleaning & distance calculation: "
        f"{df_with_distance.shape[0]} rows (Retained {len(df_with_distance)/len(raw_df)*100:.2f}%), "
        f"{df_with_distance.shape[1]} columns"
    )
    print("First rows of the dataset with distance:")
    print(df_with_distance.head())
   
    # 4. Feature Extraction, Encoding, and Robust Scaling
    final_cleaned_df, final_encoded_ml_df = process_features(df_with_distance)

    # 5. Export Data Contracts
    os.makedirs("data", exist_ok=True)
    final_cleaned_df.to_csv(CLEANED_CSV_OUT, index=False) # Exported artifact for Symbolic Rules
    final_encoded_ml_df.to_csv(ENCODED_ML_CSV_OUT, index=False) # Exported artifact for ML Training

    print("PIPELINE EXECUTION COMPLETE & CONTRACTS EXPORTED")
    print(
        f" - Cleaned CSV (Rule Engine) : {CLEANED_CSV_OUT} "
        f"({final_cleaned_df.shape[0]} Rows | {final_cleaned_df.shape[1]} Cols)"
    )
    print(
        f" - Encoded ML Matrix (Models) : {ENCODED_ML_CSV_OUT} "
        f"({final_encoded_ml_df.shape[0]} Rows | {final_encoded_ml_df.shape[1]} Cols)"
    )
    print(f" - Total Missing Values (NaN): {final_encoded_ml_df.isnull().sum().sum()}")
    print(f" - Duplicate Rows Count      : {final_encoded_ml_df.duplicated().sum()}")

    
