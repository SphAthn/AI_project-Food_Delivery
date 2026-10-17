# Student Name: Daniel Joseph Benedict Ngamelubun
# Student FAN: ngam0004
# File: src/models/ensemble_regressor.py
# Date: 18-10-2026
# Usage: Imported by src.ML_model_evaluation
# Licence: MIT Licence
# Description: Trains a Stacking Ensemble Regressor combining optimal pre-tuned XGBoost, Random Forest, Ridge, and MLP models.

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


def train_and_tune_ensemble(X_train, y_train, cv_folds=3):
    """Trains a Stacking Ensemble Regressor using optimal tuned base models.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    print("\nEnsemble Regressor (Stacking) training with optimal base hyperparameters...")

    # Define base estimators using exact best parameters
    base_estimators = [
        (
            "xgb",
            xgb.XGBRegressor(
                learning_rate=0.1,
                max_depth=9,
                n_estimators=100,
                subsample=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                max_depth=10,
                min_samples_split=5,
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0,
                solver="svd",
                random_state=42,
            ),
        ),
        (
            "mlp",
            MLPRegressor(
                activation="tanh",
                alpha=0.0001,
                hidden_layer_sizes=(128, 64),
                learning_rate_init=0.001,
                random_state=42,
                max_iter=1000,
                early_stopping=True,
            ),
        ),
    ]

    stacking_estimator = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(),
        n_jobs=-1,
    )

    param_grid = {
        "final_estimator__alpha": [0.1, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        estimator=stacking_estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Ensemble Regressor Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params
