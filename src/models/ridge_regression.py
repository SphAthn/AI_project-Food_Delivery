# Usage: python -m src.models.ridge_regression
# Description: Trains and tunes a Ridge Regressor model using GridSearchCV.

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def train_and_tune_ridge(X_train, y_train, cv_folds: int = 5):
    """Trains a Ridge Regressor using GridSearchCV.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    ridge_estimator = Ridge(random_state=42)

    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky"],
    }

    grid_search = GridSearchCV(
        estimator=ridge_estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Ridge Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params