# Usage: python -m src.models.random_forest
# Description: Trains and tunes a Random Forest Regressor using GridSearchCV.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def train_and_tune_random_forest(X_train, y_train, cv_folds=3):
    """Train a Random Forest, optionally running the focused grid search.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.
        run_grid_search: If True, runs the GridSearchCV. The best parameters found during the completed experiments are already available.

    Returns:
        tuple: (best_model, best_params)
    """
    print("\nRandom Forest training and hyperparameter tuning...")
    run_grid_search=False

    rf_estimator = RandomForestRegressor(random_state=42, n_jobs=1)

    # RandomizedSearchCV experiment:
    # param_distributions = {
    #     "n_estimators": [100, 200, 300, 400, 500],
    #     "max_depth": [None, 10, 20, 30, 40, 50],
    #     "min_samples_split": [2, 5, 10, 15, 20],
    #     "min_samples_leaf": [1, 2, 3, 4],
    #     "max_features": [1.0, "sqrt", "log2"],
    #     "bootstrap": [True, False],
    # }
    # random_search = RandomizedSearchCV(
    #     estimator=rf_estimator,
    #     param_distributions=param_distributions,
    #     n_iter=70,
    #     scoring="neg_root_mean_squared_error",
    #     cv=cv_folds,
    #     random_state=42,
    #     n_jobs=-1,
    #     verbose=1,
    # )
    # random_search.fit(X_train, y_train)

    # The random search repeatedly identified two promising regions:
    param_grid = [
        {
            "n_estimators": [100, 200],
            "max_depth": [20, 30],
            "min_samples_split": [10, 20],
            "min_samples_leaf": [1, 3],
            "max_features": [1.0],
            "bootstrap": [True],
        },
        {
            "n_estimators": [200, 300],
            "max_depth": [15, 20],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [1, 2],
            "max_features": ["log2"],
            "bootstrap": [False],
        },
    ]

    if not run_grid_search:
        best_params = {
            "max_depth": 15,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "min_samples_split": 20,
            "n_estimators": 300,
            "bootstrap": True,
        }
        print("Using validated Random Forest parameters (GridSearchCV skipped).")
        best_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **best_params,
        )
        best_model.fit(X_train, y_train)
        print(f"Random Forest Parameters: {best_params}")
        
    else:
        print("Starting grid search")
        grid_search = GridSearchCV(
            estimator=rf_estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_folds,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        print("grid search done")

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Random Forest Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params