import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

ML_csv_path: str = "data/Zomato_Encoded_ML.csv"
target_col: str = "Time_taken (min)"
test_size: float = 0.2
val_size: float = 0.15
random_state: int = 42

def get_split_scaled_data():
    """Loads raw dataset, cleans, engineers features, splits data, and scales continuous attributes without leakage.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    print("\nStarting data splitting and scaling process...")
    # 1. Load the ML dataset
    df=pd.read_csv(ML_csv_path)

    # 2. Separate Features and Target Variable
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Train / Test Split (80% Train, 20% Test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 4. Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Training set size: {X_train.shape[0]} samples")

    # 4. Fit RobustScaler on Training Set
    continuous_features = ["distance_km", "Delivery_person_Age", "Delivery_person_Ratings", "order_hour"]

    scaler = RobustScaler()
    
    # Copy to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_val[continuous_features] = scaler.transform(X_val[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    print("\nData splitting and scaling completed successfully!")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, X.columns.tolist()