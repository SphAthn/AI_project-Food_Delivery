#  Symbolic Prolog Engine & Hybrid AI Subsystem

This document details the architecture, rule mechanics, and step-by-step execution process of the **Symbolic Prolog Rule Engine** and its integration with the data-driven **Machine Learning Subsystem**.


##  System Architecture & Data Flow

```text
+------------------------------+     +-------------------------------+
|   data/Zomato_Cleaned.csv    |     | data/ML_Inference_Predictions |
| (Human-Readable Context)     |     | (Baseline ML Estimates)       |
+--------------+---------------+     +---------------+---------------+
               |                                     |
               +------------------+------------------+
                                  |
                                  v
                   +------------------------------+
                   |     src/prolog_bridge.py     |
                   |  (Data Joining & Execution)  |
                   +--------------+---------------+
                                  |
                                  v
                   +------------------------------+
                   |  prolog/dispatch_rules.pl    |
                   |  (Symbolic Rule Engine)      |
                   +--------------+---------------+
                                  |
                                  v
                   +------------------------------+
                   | data/Hybrid_Inference_Outputs|
                   | (Final Hybrid ETAs & Flags)  |
                   +------------------------------+
```


##  Step-by-Step Execution Process

### Step 1: Human-Readable Context & ML Prediction Coupling
* **Inputs:** `data/Zomato_Cleaned.csv` and `data/ML_Inference_Predictions.csv`
* **Process:** The bridge script (`src/prolog_bridge.py`) aligns the isolated test set predictions from the winning ML model (e.g., Stacking Ensemble) with the original unencoded contextual metadata (`Weather_conditions`, `Road_traffic_density`, `Delivery_person_Ratings`, `distance_km`, `Festival`, `multiple_deliveries`).


### Step 2: Symbolic Knowledge Base Rule Evaluation (`prolog/dispatch_rules.pl`)
Each test record is evaluated against declarative logic rules in SWI-Prolog across two major categories:

#### 1. Operational Guardrails & Dispatch Status
* **`REJECT_DISTANCE_EXCEEDED`**: Applied if delivery distance $> 25.0\text{ km}$.
* **`RISK_LOW_RATING_PEAK_TRAFFIC`**: Flagged if courier rating $< 3.8$ during `'Jam'` or `'High'` traffic density.
* **`RISK_MULTIPLE_DELIVERIES_BAD_WEATHER`**: Flagged if handling multiple deliveries during severe weather (`Stormy`, `Sandstorms`, `Fog`).
* **`APPROVED`**: Default status for safe, standard dispatches.

#### 2. Domain-Specific Penalty Adjustments (Minutes)
* **Weather Penalties:** `Stormy` ($+15\text{ min}$), `Sandstorms` ($+12\text{ min}$), `Fog` ($+8\text{ min}$), `Windy` ($+5\text{ min}$), `Cloudy` ($+3\text{ min}$).
* **Traffic Penalties:** `Jam` ($+12\text{ min}$), `High` ($+7\text{ min}$), `Medium` ($+3\text{ min}$).
* **Festival Surge:** `Yes` ($+15\text{ min}$).
* **Multi-Delivery Load:** $+4\text{ min}$ per active delivery load.

---

### Step 3: Inference Bridge Execution (`src/prolog_bridge.py`)
* **Process:** Uses Python's `subprocess` module to query `swipl` (SWI-Prolog CLI) in headless mode.
* **Query Format:**
  ```prolog
  evaluate_order(MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDeliveries, [FinalEta, Penalty, Flag])
  ```
* **Computation:**
  $$\text{Final\_Hybrid\_ETA} = \text{Predicted\_ETA\_ML} + \text{Total\_Prolog\_Penalty}$$



### Step 4: Output Contract Export & Validation Metrics
* **Output Artifact:** `data/Hybrid_Inference_Outputs.csv`
* **Evaluation:** Calculates overall Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$) comparing:
  1. Ground-Truth Actual Time (`Time_taken (min)`) vs. Baseline ML Estimates (`Predicted_ETA_ML(min)`)
  2. Ground-Truth Actual Time vs. Hybrid AI Final Predictions (`Hybrid_Final_ETA(min)`)



##  How to Run Prolog Evaluation

### Prerequisites
Ensure **SWI-Prolog** is installed on your operating system and added to your system PATH:
```bash
swipl --version
```

### Execution Command
Run the bridge script directly from the root project folder:
```bash
python -m src.prolog_bridge
```
