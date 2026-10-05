# Zomato Delivery Pipeline (COMP9742 - Artefact 1)

Automatic data ingestion pipeline for the Food-Delivery team project. It downloads the Zomato Delivery Operations dataset, audits and cleans it, computes the distance between the restaurant and the customer, encodes and scales the features, and exports two CSV files. 

This is the preprocessing phase (Artefact 1) for the real project: a machine learning model that predicts delivery time, and a symbolic rule engine that adjusts the prediction with safety rules.

Git Repository: https://github.com/SphAthn/AI_project-Food_Delivery.git


## Project structure

```text
main.py                      # Runs the whole pipeline (5 steps)
requirements.txt             # Python packages
README.md
src/
├──__init__.py
├──load_dataset.py            
├──missing_values.py         
├──distance.py                
└──feature_engineering.py 
data/                        # Created automatically, the final CSV files are saved here
```

## Requirements

- Python 3.11 or newer
- The packages listed in `requirements.txt`
- An internet connection to download the Kaggle dataset

## How to run

Run these commands from the project folder:

```text
pip install -r requirements.txt   # if needed
python main.py
```

No dataset file is needed. The data is downloaded automatically from KaggleHub (`saurabhbadole/zomato-delivery-operations-analytics-dataset`).




The console shows the 5 steps of the pipeline:

1. **Load data** (`load_dataset.py`): downloads the CSV and shows its size.
2. **Clean data** (`missing_values.py`): prints the anomalies found, then removes or repairs them and reports the result.
3. **Compute distance** (`distance.py`): adds the column `distance_km`.
4. **Feature engineering** (`feature_engineering.py`): extracts the order hour and applies one-hot encoding.
5. **Export**: writes the two CSV files and prints a summary (rows, columns, missing values, duplicates, total time).


## Outputs (data contract)

| File | Content | Used by |
|---|---|---|
| `data/Zomato_Cleaned.csv` | Clean and readable | Symbolic rule engine |
| `data/Zomato_Encoded_ML.csv` | Numbers only: identifiers, raw dates and raw GPS removed | Machine learning model |

## Settings

The distance method is chosen in `main.py` in the call
`compute_distance(clean_df, method="manhattan")`. The values `"haversine"` and
`"euclidean"` are also available.
