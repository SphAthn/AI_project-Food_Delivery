# Usage: python -m src.models.neural_network
# Description: Trains and tunes a Multi-Layer Perceptron (MLP) Neural Network using GridSearchCV.

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def train_and_tune_neural_network(X_train, y_train, cv_folds: int = 3):
    """Trains a Multi-Layer Perceptron (MLP) Regressor using GridSearchCV.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    mlp_estimator = MLPRegressor(
        random_state=42,
        max_iter=500,
        early_stopping=True,
    )

    param_grid = {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }

    grid_search = GridSearchCV(
        estimator=mlp_estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"MLP Neural Network Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params