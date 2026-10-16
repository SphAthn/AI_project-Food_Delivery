# Usage: python -m src.models.random_forest
# Description: Trains and tunes a Random Forest Regressor using GridSearchCV.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def train_and_tune_random_forest(X_train, y_train, cv_folds: int = 3):
    """Trains a Random Forest Regressor using GridSearchCV.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    rf_estimator = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=rf_estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Random Forest Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params