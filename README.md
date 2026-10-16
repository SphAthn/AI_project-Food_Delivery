
## Overview
This repository implements a modular Hybrid AI Prediction and Inference Engine for urban food delivery duration estimation. It combines a data-driven Machine Learning regression pipeline with a symbolic Prolog rule engine to enforce real-world dispatch safety guardrails and weather/traffic penalty adjustments.

---

## Directory Structure
```text
A2_ngam0004/
├── Reports/
│   ├── ngam0004_A2_Report.pdf
│   └── ngam0004_A2_SelfAssessment.pdf
└── Source/
    ├── main.py
    ├── README.md
    ├── requirements.txt
    └── src/
        ├── __init__.py
        ├── load_dataset.py
        ├── missing_values.py
        ├── distance.py
        ├── feature_engineering.py
        ├── data_preprocessing.py
        ├── evaluate_models.py
        └── models/
            ├── __init__.py
            ├── ridge_regression.py
            ├── random_forest.py
            └── xgboost_model.py
            └── neural_network.py


Source/
│
├── main.py                     # Master execution script (Runs preprocessing, model training, evaluation, and Prolog bridge)
├── requirements.txt            # Package dependencies (pip freeze)
├── README.md                   # Execution instructions, team GitHub link, bug list
│
├── src/                        # Modular Python Package
│   ├── __init__.py
│   ├── load_dataset.py         # KaggleHub + local fallback ingestion
│   ├── missing_values.py       # Cleaning, filtering (<18, GPS<=0), median/Unknown imputation
│   ├── distance.py             # Vectorized Manhattan distance calculation
│   ├── feature_engineering.py  # Order hour parsing & One-Hot Encoding (UNSCALED)
│   ├── data_preprocessing.py   # Ingestion orchestrator + Train/Val/Test Split + Scaler Pipeline
│   │
│   ├── models/                 
│   │   ├── __init__.py
│   │   ├── linear_regression.py # Ridge/Lasso/Linear Regression with GridSearchCV
│   │   ├── random_forest.py     # Random Forest Regressor with GridSearchCV
│   │   ├── xgboost_model.py     # XGBoost Regressor with GridSearchCVs
│   │   └── neural_network.py    # MLP Regressor (PyTorch or Scikit-Learn)
│   │
│   ├── evaluate_models.py      # Compare all models on Validation Set (RMSE, MAE, R2) & export best .pkl
│   └── prolog_bridge.py        # PySWIP / Subprocess bridge to pass ML predictions to Prolog (later)
│
└── prolog/                     # Symbolic Rule Engine
    └── dispatch_rules.pl       # Knowledge Base (facts, safety guardrails, weather/traffic ETA penalties)