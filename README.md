# Overview

This repository implements a modular Hybrid AI Prediction and Inference Engine for urban food delivery duration estimation. It combines a data-driven Machine Learning regression pipeline with a symbolic Prolog rule engine to enforce real-world dispatch safety guardrails and weather/traffic penalty adjustments.

Git Repository: https://github.com/SphAthn/AI_project-Food_Delivery.git


## Project structure

```text
AI_project-Food_Delivery/
│
├── main.py                     # Master execution pipeline
├── inspect_model.py            # Model inspection & hyperparameter utility
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
│
├── src/                        # Modular Machine Learning Subsystem
│   ├── __init__.py
│   ├── data_preprocessing.py   # Ingestion orchestrator & pipeline manager
│   ├── split_scale.py          # Leakage-free split & RobustScaler transformation
│   ├── ML_model_evaluation.py  # Model hyperparameter tuning & validation benchmark
│   ├── ML_predictions.py       # Test set prediction evaluation
│   │
│   ├── preprocessing/          # Feature engineering & data hygiene submodules
│   │   ├── distance.py         # Coordinate distance calculation (Manhattan)
│   │   ├── EDA_visualisation.py# Exploratory data analysis scripts
│   │   ├── feature_engineering.py # Temporal extraction & one-hot encoding
│   │   ├── load_dataset.py     # Ingestion via KaggleHub API with local fallback
│   │   └── missing_values.py   # Data sanitization, filtering & median imputation
│   │
│   └── models/                 # Model tuning specifications (GridSearchCV)
│       ├── __init__.py
│       ├── ridge_regression.py # Ridge Regression tuner (
│       ├── random_forest.py    # Random Forest Regressor tuner (Sophie)
│       ├── xgboost_model.py    # XGBoost Regressor tuner (daniel)
│       ├── neural_network.py   # MLP Neural Network tuner
│       └── ensemble_regressor.py # Stacking Ensemble Regressor tuner (daniel)
│
└── data/                       # Generated Artifacts & Data Contracts (Auto-created)
    ├── Zomato_Cleaned.csv      # Unencoded cleaned dataset (Rule Engine contract)
    ├── Zomato_Encoded_ML.csv   # One-hot encoded dataset (Machine Learning contract)
    ├── best_ml_model.pkl       # Serialized winning ML model (Joblib binary)
    └── ML_Inference_Predictions.csv # Test set predictions + actuals (Prolog contract)

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

##  Output Artifacts

Running `main.py` generates 4 essential output files inside `data/`:

| File Name | Format | Shape / Size | Purpose & Description |
| :--- | :--- | :--- | :--- |
| **`Zomato_Cleaned.csv`** | CSV | 41,488 rows × 22 cols | Cleaned human-readable dataset. Serves as the primary data contract for Prolog symbolic rule reasoning. |
| **`Zomato_Encoded_ML.csv`** | CSV | 41,488 rows × 34 cols | One-hot encoded numeric matrix used for machine learning model training. |
| **`best_ml_model.pkl`** | Binary `.pkl` | Model Object | Serialized binary object of the winning Stacking Ensemble Regressor exported via `joblib`. |
| **`ML_Inference_Predictions.csv`** | CSV | 8,298 rows × 35 cols | Evaluation matrix containing test features, ground-truth actual times (`Actual_Time_taken(min)`), and ML baseline estimates (`Predicted_ETA_ML(min)`). |

---

## Step-by-Step Execution Process

1. **Automated Data Ingestion & Hygiene:** Downloads raw data via `kagglehub`, filters invalid GPS coordinates (≤ 0) and underage couriers (< 18), and imputes numerical nulls using training medians.
2. **Feature Engineering & Coordinate Math:** Computes Manhattan delivery distance (`distance_km`) and extracts temporal order features (`order_hour`).
3. **Leakage-Free Splitting & Scaling:** Splitting occurs prior to scaling (68% Train / 12% Val / 20% Test). Continuous variables are scaled using `RobustScaler` fit strictly on the training set.
4. **GridSearchCV Hyperparameter Tuning:** Cross-validates 5 or more distinct model architectures and ranks performance by RMSE.
5. **Inference & Artifact Serialization:** Saves the winning model (`best_ml_model.pkl`) and exports prediction matrices for Prolog symbolic rule reasoning.
