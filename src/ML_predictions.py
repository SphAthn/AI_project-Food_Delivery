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

    # Export predictions CSV contract for Prolog engine
    os.makedirs("data", exist_ok=True)

    # Convert X_test to DataFrame with feature names if present
    if isinstance(X_test, pd.DataFrame):
        inference_df = X_test.copy()
    elif feature_names is not None:
        inference_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        inference_df = pd.DataFrame(X_test)

    # Attach targets and predicted values
    if hasattr(y_test, "values"):
        inference_df["Actual_Time_taken(min)"] = y_test.values
    else:
        inference_df["Actual_Time_taken(min)"] = np.array(y_test)

    inference_df["Predicted_ETA_ML(min)"] = np.round(predictions, 2)

    out_csv = "data/ML_Inference_Predictions.csv"
    inference_df.to_csv(out_csv, index=False)
    print(f" - Exported ML predictions artifact to '{out_csv}'")

    return predictions

