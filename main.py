# File: main.py
# Date: 08-10-2026
# Description: Runs the full data ingestion pipeline and exports  CSV files.
# Usage: python main.py

from src.data_preprocessing import get_prepared_data


# Parameters
preprocessing_needed = True  # Set to True if you want to run the preprocessing pipeline and generate CSVs


if __name__ == "__main__":
    print("Starting Pipeline Execution...")

    if preprocessing_needed:
        get_prepared_data()

    print("\nPipeline execution completed successfully!")

