# File: analysis/penalty_calibration.py
# Usage (from the project ROOT): python analysis/penalty_calibration.py
# Description: Measures how much the ML model still under/over-estimates delivery time for each condition, to calibrate the Prolog penalties.
# Results are used in prolog/dispatch_rules.pl file

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/Hybrid_Inference_Outputs.csv")
preds = pd.read_csv("data/ML_Inference_Predictions.csv")

df["actual"] = preds["Actual_Time_taken(min)"].values
df["residual"] = df["actual"] - df["Predicted_ETA_ML(min)"].values   # > 0 : ML too optimistic

# 1. raw group means
print("=== Mean residual per category (minutes) ===\n")
for col in ["Weather_conditions", "Road_traffic_density", "Festival", "multiple_deliveries"]:
    print(df.groupby(col)["residual"].agg(["mean", "count"]).round(2), "\n")

# 2. additive coefficients
# Group means overlap (a Jam day is often also a Stormy day), so summing them double-counts.
# A linear regression gives the additive effect of each factor, which is exactly how
# the Prolog penalties are combined (sum of penalties).
REFERENCE = ["Weather_conditions_Sunny", "Road_traffic_density_Low", "Festival_No"]

X = pd.get_dummies(df[["Weather_conditions", "Road_traffic_density", "Festival"]], dtype=float)
X = X.drop(columns=REFERENCE, errors="ignore")
X["multiple_deliveries"] = df["multiple_deliveries"].astype(float)

# Calibrate on a random half, check on the other half (avoids grading the rules on the
# same data they were fitted on).
calib = df.sample(frac=0.5, random_state=42)
hold = df.drop(calib.index)

A = np.column_stack([np.ones(len(calib)), X.loc[calib.index].values])
coef, *_ = np.linalg.lstsq(A, calib["residual"].values, rcond=None)
coef_s = pd.Series(coef, index=["intercept"] + list(X.columns))

print("=== Additive penalties (regression on calibration half, minutes) ===")
print(coef_s.round(2), "\n")
print("Intercept = residual of the reference case (Sunny / Low traffic / no festival / 0 deliveries).")
print("multiple_deliveries = minutes PER extra delivery.\n")

# 3. holdout check
adj = X.loc[hold.index].values @ coef[1:]
ml = hold["Predicted_ETA_ML(min)"].values
hyb = ml + np.round(adj * 2) / 2 # penalties rounded to 0.5 min, as in the .pl file
y = hold["actual"].values

def rmse(a, b): return mean_squared_error(a, b) ** 0.5

print(f"=== Holdout check (n = {len(hold)}) ===")
print(f"ML only     : MAE {mean_absolute_error(y, ml):.3f} | RMSE {rmse(y, ml):.3f}")
print(f"ML + penalty: MAE {mean_absolute_error(y, hyb):.3f} | RMSE {rmse(y, hyb):.3f}")


print("=== Spread and 80th percentile of residuals per category ===\n")
for col in ["Weather_conditions", "Road_traffic_density", "Festival", "multiple_deliveries"]:
    print(df.groupby(col)["residual"]
            .agg(mean="mean", std="std", q80=lambda s: s.quantile(0.8), n="count")
            .round(2), "\n")

print("Global q80:", round(df["residual"].quantile(0.8), 2))
print("Late rate, ML only:", round((df["actual"] > df["Predicted_ETA_ML(min)"]).mean(), 3))


print("=== Safety buffer trade-off ===")
for b in [0, 1, 2, 3, 3.5]:
    late = (df["actual"] > df["Predicted_ETA_ML(min)"] + b).mean()
    mae = (df["actual"] - (df["Predicted_ETA_ML(min)"] + b)).abs().mean()
    print(f"buffer {b:>3} min | late rate {late:.1%} | MAE {mae:.2f}")