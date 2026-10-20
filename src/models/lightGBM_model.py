# Usage: python -m src.models.lightgbm_model
# Description: Trains and tunes a LightGBM Regressor model using GridSearchCV.

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


def train_and_tune_lightgbm(X_train, y_train, cv_folds: int = 3):
    """Trains a LightGBM Regressor using GridSearchCV.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    lgb_estimator = lgb.LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbose=-1,  # silence LightGBM's per-fit logging
    )

    param_grid = {
        "n_estimators": [100, 200],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=lgb_estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"LightGBM Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params