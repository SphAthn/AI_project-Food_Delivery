"""
Main entry point for the project. 

Run with:
    python main.py

"""

# Configuration and constants
RAW_CSV_NAME = "data/Zomato_Dataset_raw.csv" 


from src.distance import compute_distance
from src.load_dataset import load_dataset



if __name__ == "__main__":
    # Load the raw dataset
    raw_df = load_dataset(RAW_CSV_NAME)

    print("Loaded raw dataset:", raw_df.shape[0], " rows, ", raw_df.shape[1], " columns")
    print("First rows of the dataset:")
    print(raw_df.head())

    # Check for missing values
    # ...

    # Compute distance
    df_with_distance = compute_distance(raw_df, method="manhattan")

    print("\nDataset with distance: ", df_with_distance.shape[0], " rows, ", df_with_distance.shape[1], "columns")
    print("First rows of the dataset with distance:")
    print(df_with_distance.head())

    # Other preprocessing steps
    #...

    final_df = df_with_distance

    

    # Save the preprocessed dataset to a new CSV file
    final_df.to_csv("data/Zomato_Dataset_preprocessed.csv", index=False)
