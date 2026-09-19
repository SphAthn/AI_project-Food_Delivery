# AI_project-Food_Delivery
# Zomato Delivery Operations Analytics Pipeline

An automated, fault-tolerant Data Ingestion, Preprocessing, and Feature Engineering Pipeline. Built for the Zomato Delivery Operations dataset, this pipeline ingests raw delivery telemetry and outputs clean, high-dimensional representations to support a Neuro-Symbolic Hybrid AI Engine.

## Repository Structure

```text
zomato-pipeline/
├── data/
│   ├── Zomato Dataset.csv              # Optional local fallback dataset
│   ├── Zomato_Cleaned.csv              # Exported artifact for Symbolic Rules
│   └── Zomato_Encoded_ML.csv           # Exported artifact for ML Training
├── src/
│   ├── __init__.py
│   ├── load_dataset.py                 # KaggleHub + Local Fallback Ingestion
│   ├── missing_values.py               # Sanitization, Filtering, and Imputation
│   ├── distance.py                     # Vectorized Haversine Distance Engine
│   └── feature_engineering.py          # Temporal Extraction & RobustScaler
├── main.py                             # Central Execution Script
└── requirements.txt                    # Project Python Dependencies (have not been created)
