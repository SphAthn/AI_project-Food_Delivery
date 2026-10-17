# Usage: python -m src.models.neural_network
# Description: Trains and tunes a Neural Network using GridSearchCV.


from sklearn.model_selection import GridSearchCV



def train_and_tune_neural_network(X_train, y_train, cv_folds=3):
    """Trains a Neural Network using GridSearchCV.

    Args:
        X_train: Scaled training features DataFrame or array.
        y_train: Training target Series or array.
        cv_folds: Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params)
    """
    print("\nNeural Network training and hyperparameter tuning...")
    rf_estimator = ()

    param_grid = {}

    grid_search = GridSearchCV(estimator=rf_estimator, param_grid=param_grid, scoring= "neg_mean_squared_error",
        cv=cv_folds, n_jobs=-1, verbose=1)
   
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Neural Network Tuning Complete. Best Parameters: {best_params}")
    return best_model, best_params