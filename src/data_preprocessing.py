import os

from src.preprocessing.distance import compute_distance # Distance Calculation (Euclidean, Manhattan or Haversine)
from src.preprocessing.feature_engineering import process_features # Temporal Extraction & RobustScaler
from src.preprocessing.load_dataset import load_dataset # KaggleHub + Local Fallback Ingestion
from src.preprocessing.missing_values import clean_dataset # Sanitization, Filtering, and Imputation


# Configuration and Constants
RAW_CSV_NAME = "data/Zomato Dataset.csv"
CLEANED_CSV_OUT = "data/Zomato_Cleaned.csv" # Exported artifact for Symbolic Rules
ENCODED_ML_CSV_OUT = "data/Zomato_Encoded_ML.csv" # Exported artifact for ML Training




def get_prepared_data():
    print("Preprocessing the raw dataset...")

    # 1. Load the Raw Dataset
    print("\nStep 1: Load dataset")
    raw_df = load_dataset(RAW_CSV_NAME)
    print(f"Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns\n")
    print("First rows of the dataset:")
    print(raw_df.head())
    
    # 2. Clean Dataset and Perform NaN/Anomaly Audit
    print("\nStep 2: Clean dataset and handle missing values")
    clean_df = clean_dataset(raw_df)

    # 3. Compute Distance (km)
    print("\nStep 3: Compute distance (km) for each order")
    df_with_distance = compute_distance(clean_df, method="manhattan") #or euclidean or haversine

    print(
        f"\nDataset after cleaning & distance calculation: "
        f"{df_with_distance.shape[0]} rows (Retained {len(df_with_distance)/len(raw_df)*100:.2f}%), "
        f"{df_with_distance.shape[1]} columns"
    )
    print("First rows of the dataset with distance:")
    print(df_with_distance.head())
   
    # 4. Feature Extraction, Encoding, and Robust Scaling
    print("\nStep 4: Feature extraction and encoding")
    final_cleaned_df, final_encoded_ml_df = process_features(df_with_distance)

    # 5. Export Data Contracts
    print("\nStep 5: Export dataset to CSV")
    os.makedirs("data", exist_ok=True)
    final_cleaned_df.to_csv(CLEANED_CSV_OUT, index=False) # Exported artifact for Symbolic Rules
    final_encoded_ml_df.to_csv(ENCODED_ML_CSV_OUT, index=False) # Exported artifact for ML Training



    print("\nPIPELINE EXECUTION COMPLETE")
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


if __name__ == "__main__":
    get_prepared_data()
