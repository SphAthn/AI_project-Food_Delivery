"""
Main entry point for the project. 

Run with:
    python main.py

"""

# Configuration and constants
RAW_CSV_NAME = "data/Zomato_Dataset_raw.csv" 


from src.load_dataset import load_dataset



if __name__ == "__main__":
    # Load the raw dataset
    raw_df = load_dataset(RAW_CSV_NAME)
    
    print(f"Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(raw_df.head())
