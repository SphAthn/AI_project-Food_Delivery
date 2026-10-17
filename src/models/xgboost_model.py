# Student Name: Daniel Joseph Benedict Ngamelubun
# Student FAN: ngam0004
# File: src/models/xgboost_model.py
# Date: 18-10-2026
# Usage: Imported by src.ML_model_evaluation, python -m src.models.xgboost_model
# Description: Trains and tunes an XGBoost Regressor model using GridSearchCV.

import xgboost as xgb
from sklearn.model_selection import GridSearchCV


def train_and_tune_xgboost(X_train, y_train, cv_folds=3):
    """Trains a XGBoost model using GridSearchCV.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    print("\nXGBoost training and hyperparameter tuning...")

    xgb_estimator = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"XGBoost Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params
