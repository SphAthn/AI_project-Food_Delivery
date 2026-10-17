# Usage: python -m src.evaluate_models
# Description: Evaluates and compares all tuned machine learning models on a validation set and saves the winning model.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#from src.models.neural_network import train_and_tune_neural_network
from src.models.random_forest import train_and_tune_random_forest
#from src.models.ridge_regression import train_and_tune_ridge
#from src.models.xgboost_model import train_and_tune_xgboost


def evaluate_model_performance(model, X_val, y_val):
    """Calculates RMSE, MAE, and R2 performance metrics on the validation set."""
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    return rmse, mae, r2


def run_model_comparison(X_train, X_val, y_train, y_val):
    """Runs the model comparison process. 
    Tunes the hyperparameters of each model
    Evaluates the best models on the validation set
    Keeps the best performing model."""

    models_to_evaluate = {
#        "Ridge Regression": train_and_tune_ridge,
        "Random Forest": train_and_tune_random_forest,
#        "XGBoost Regressor": train_and_tune_xgboost,
#        "MLP Neural Network": train_and_tune_neural_network,
    }

    results = []
    trained_models = {}

    print("\nFinding the best hyperparameter for each model and evaluating on the validation set")
    for model_name, train_fn in models_to_evaluate.items():
        best_model, best_params = train_fn(X_train, y_train)
        
        # Evaluate on Validation Set
        rmse, mae, r2 = evaluate_model_performance(best_model, X_val, y_val)
        
        results.append({
            "Model": model_name,
            "Validation RMSE (min)": round(rmse, 4),
            "Validation MAE (min)": round(mae, 4),
            "Validation R2 Score": round(r2, 4),
            "Best Parameters": str(best_params),
        })
        trained_models[model_name] = best_model

    # Convert results to DataFrame
    comparison_df = pd.DataFrame(results).sort_values(by="Validation RMSE (min)", ascending=True)

    print("            MODEL EVALUATION COMPARISON TABLE           ")
    print(comparison_df.to_string(index=False))

    # Select Winning Model (Lowest RMSE)
    winning_model_name = comparison_df.iloc[0]["Model"]
    winning_model = trained_models[winning_model_name]

    return winning_model_name, winning_model
    
if __name__ == "__main__":
    run_model_comparison()