# Technical Code Explanation & Data Pipeline Documentation

**Author:** Daniel Joseph Benedict Ngamelubun  
**Student FAN:** ngam0004  
**File Name:** codeexplanation.md and Daniel work-A1.ipynb
**Date:** 16-09-2026  
**Project:** Zomato Delivery Operations Analytics — Ingestion & Preprocessing Subsystem  


## 1. Executive Summary & Pipeline Overview

The primary goal of this module is to establish an automated, fault-tolerant Data Ingestion and Preprocessing Pipeline. Operating on the **Zomato Delivery Operations Analytics** dataset, this pipeline ingests raw, unstructured delivery telemetry and transforms it into clean, high-dimensional representations for downstream modeling.

To support a **Neuro-Symbolic Hybrid AI Engine**, the pipeline splits output data into two distinct artifacts:
1. **'Zomato_Cleaned.csv'**: A human-readable dataset designed for deterministic symbolic logic, rule evaluation, and expert system guardrails.
2. **'Zomato_Encoded_ML.csv'**: A fully dummy-encoded, robustly scaled numeric matrix built specifically for machine learning regression algorithms.


## 2. Technical Data Pipeline Breakdown

### Step 1: Automated Data Ingestion & Fallback Audit
- **Methodology:** The ingestion module utilizes the 'kagglehub' Python API ('saurabhbadole/zomato-delivery-operations-analytics-dataset') to fetch the primary dataset directly into local memory.
- **Fault-Tolerant Fallback:** If API access fails due to network restrictions or missing API keys, the pipeline falls back to loading a local copy ('Zomato Dataset.csv').
- **Raw Audit Metrics:** The raw dataset comprises **45,584 rows** and **20 features**, containing a mix of numerical, categorical, temporal, and spatial coordinates.


### Step 2: Data Sanitization, Anomaly Filtering & Imputation

| Metric / Anomaly Category | Raw Issue Identified | Action Taken & Pipeline Logic | Retained Impact |
| :--- | :--- | :--- | :--- |
| **Driver Rating Errors** | 53 records contained ratings of '6.0' (exceeding the 5.0 scale cap). | Capped all rating values $> 5.0$ to exactly '5.0'. | Corrected distribution bias. |
| **Invalid GPS Coordinates** | 4,071 records (8.93%) contained coordinates $\le 0.0$. | Filtered out all invalid coordinate rows to prevent distance calculation corruption. | Preserved spatial integrity. |
| **Underage Couriers** | 38 records listed driver ages $< 18$ years old. | Filtered out underage records to enforce operational compliance. | Enforced legal bounds. |
| **String Formatting & Nulls** | Categorical strings contained whitespace and irregular "NaN" labels. | Stripped trailing/leading whitespace across text columns. Imputed missing driver ages/ratings using median values, and missing categoricals with "Unknown". | Achieved 0 missing values. |

- **Final Cleaned Retention:** The sanitization process yields **40,012 clean operational records** (an **87.78% retention rate**) with **0 duplicate rows** and **0 missing values ('NaN')**.



### Step 3: Spatial & Temporal Feature Engineering

#### A. Haversine Spatial Distance ('distance_km')
Standard Euclidean distance fails on geographical coordinates because it ignores Earth's curvature. The pipeline implements the **Haversine Formula** to compute the true great-circle distance (in kilometers) between restaurant coordinates $(\phi_1, \lambda_1)$ and delivery destination coordinates $(\phi_2, \lambda_2)$:

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1) \cdot \cos(\phi_2) \cdot \sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \arctan2\left(\sqrt{a}, \sqrt{1-a}\right)$$

$$d = R \cdot c$$

*(where $R = 6371\text{ km}$, Earth's radius)*

#### B. Dispatch Order Hour ('order_hour')
Extracts the integer hour (0–23) from the timestamp string 'Time_Orderd'. This temporal feature captures daily traffic surge patterns, peak meal delivery hours (lunch/dinner rushes), and driver availability shifts.


### Step 4: Categorical Encoding & Robust Feature Scaling

#### A. One-Hot Encoding
Categorical variables ('Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City') are converted into sparse binary indicator columns ($0$ or $1$) using 'pd.get_dummies()'. This expands the feature space from 22 columns to 40 columns, eliminating implicit ordinal bias.

#### B. Robust Feature Scaling ('RobustScaler')
Continuous numeric variables ('distance_km', 'Delivery_person_Age', 'Delivery_person_Ratings', 'order_hour') are scaled using 'sklearn.preprocessing.RobustScaler'. Unlike 'StandardScaler' or 'MinMaxScaler', `RobustScaler` uses the median and Interquartile Range (IQR):

$$X_{\text{scaled}} = \frac{X - \text{Median}(X)}{\text{IQR}(X)}$$

- **Why 'RobustScaler'?** It prevents extreme delivery distances or rare driver rating values from distorting model gradient updates, ensuring robust neural and tree-based model convergence.
- **Discrete Metric Preservation:** Discrete count features such as 'multiple_deliveries' are excluded from continuous scaling to preserve their natural zero-indexed drop counts ($0, 1, 2, 3$).



## 3. Purpose & Allocation of Output Artifacts

### 1. 'Zomato_Cleaned.csv' (22 Columns)
- **Characteristics:** Contains human-readable categorical strings ("Fog", "Jam", "Metropolitian"), raw timestamps, parsed 'order_hour', and unscaled Haversine distances in kilometers.
- **Primary Use Case:** Serves as the primary input feed for the downstream **Symbolic Reasoning Engine / Expert System Guardrails** (OWL, Datalog, Prolog, or Python Rule Engines). Symbolic engines evaluate logical constraints, driver safety rules, and dispatch policy overrides.

### 2. 'Zomato_Encoded_ML.csv' (40 Columns)
- **Characteristics:** Purely numeric, matrix-ready tensor containing 0 missing values. All nominal strings are converted to One-Hot encoded binary indicators, and continuous features are scaled via 'RobustScaler'.
- **Primary Use Case:** Serves as the direct feature matrix for **Machine Learning Regression Models** (Decision Tree Regressor, Random Forest, XGBoost, LightGBM, Linear/Ridge Regression) to predict continuous delivery durations ('Time_taken (min)').


## 4. Why Use 'Zomato_Encoded_ML.csv' for Regression?

1. **Algorithm Compatibility:** Standard machine learning algorithms operate strictly on mathematical vectors. They cannot process raw string categories like "Jam" or "Stormy". Dummy encoding provides binary numeric columns suitable for matrix multiplication and tree splits.
2. **Gradient Stability:** Continuous attributes like distance and driver age operate on different numeric scales. 'RobustScaler' normalizes these features to a median of 0 and an IQR of 1, preventing large scale features from dominating model loss calculations.
3. **Optimized Loss Convergence:** Pre-engineered spatial distances ('distance_km') and temporal dispatch hours ('order_hour') supply regression algorithms with highly predictive linear signals, maximizing out-of-sample $R^2$ performance.