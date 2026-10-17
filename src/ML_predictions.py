# Usage: python -m src.evaluate_models
# Description: Evaluates and compares all tuned machine learning models on a validation set and saves the winning model.



import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_predictions(X_test, y_test, best_model):
    """Runs predictions on the test set using the best model and evaluates performance metrics."""
    print("\nRunning predictions on the test set using the best model...")
    predictions = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nTest Set Performance Metrics:")
    print(f" - RMSE (min): {rmse:.4f}")
    print(f" - MAE (min): {mae:.4f}")
    print(f" - R2 Score: {r2:.4f}")

    return predictions 

