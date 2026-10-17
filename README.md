# Overview

This repository implements a modular Hybrid AI Prediction and Inference Engine for urban food delivery duration estimation. It combines a data-driven Machine Learning regression pipeline with a symbolic Prolog rule engine to enforce real-world dispatch safety guardrails and weather/traffic penalty adjustments.

Git Repository: https://github.com/SphAthn/AI_project-Food_Delivery.git


## Project structure

```text
Source/
│
├── main.py                     # Runs the whole pipeline
├── requirements.txt            # Python packages
├── README.md                   
│
├── src/                        
│   ├── __init__.py
│   │
│   ├── data_preprocessing.py   # Ingestion orchestrator + Train/Val/Test Split + Scaler Pipeline
│   ├── preprocessing/          # contains sub_modules of the preprocessing module
│   │   ├──distance.py
│   │   ├──EDA_visualisation.py
│   │   ├──feature_engineering.py
│   │   ├──load_dataset
│   │   └──missing_values.py
│   │
│   ├── evaluate_models.py      # Compare all models on Validation Set (RMSE, MAE, R2)
│   ├── models/                 
│   │   ├── __init__.py
│   │   ├── ridge_regression.py  # Ridge/Lasso/Linear Regression with GridSearchCV
│   │   ├── random_forest.py     # Random Forest Regressor with GridSearchCV
│   │   ├── xgboost_model.py     # XGBoost Regressor with GridSearchCVs
│   │   └── neural_network.py    # MLP Regressor (PyTorch or Scikit-Learn)
│   ├── ML_predictions.py        # Predictions from the best model found
│   │
│   ├── split_scale.py           # After the preprocessing, we split and scale the dataset to be able to perform predictions
│   └──data/                     # Created automatically, the final CSV files are saved here


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
