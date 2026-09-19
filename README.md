# AI_project-Food_Delivery
zomato-pipeline/
├── data/
│   ├── Zomato Dataset.csv              # Optional local fallback dataset
│   ├── Zomato_Cleaned.csv              # Exported artifact for Symbolic Rules
│   └── Zomato_Encoded_ML.csv           # Exported artifact for ML Training
├── src/
│   ├── __init__.py
│   ├── load_dataset.py                 # KaggleHub + Local Fallback Ingestion
│   ├── missing_values.py               # Sanitization, Filtering, and Imputation
│   ├── distance.py                     # Vectorized Haversine Distance Engine /or manhattan
│   └── feature_engineering.py          # Temporal Extraction & RobustScaler
├── main.py                             # Execution Script
└── requirements.txt                    # have not been created yet
