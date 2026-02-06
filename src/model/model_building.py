import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.logger import logging


# =========================================================
# 1️⃣ Load Data
# =========================================================
def load_data(train_path: str, test_path: str):

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    return train_df, test_df


# =========================================================
# 2️⃣ Feature Selection
# =========================================================
def select_features(train_df, test_df):

    input_columns = (
        ["SOC", "SOH", "Battery_pack_total_voltage", "Battery_current"]
        + [f"Battery_{i}_Volt" for i in range(1, 17)]
        + [f"temperature_{i}" for i in range(1, 7)]
    )

    target_columns = [
        f"temperature_{i}" for i in range(1, 7) if i != 2
    ]

    train_df = train_df[input_columns].dropna().reset_index(drop=True)
    test_df  = test_df[input_columns].dropna().reset_index(drop=True)

    return train_df, test_df, target_columns


# =========================================================
# 3️⃣ Sliding Window Creation
# =========================================================
def create_sliding_windows(data, target_cols, W=30, H=30):

    X, y = [], []
    target_idx = [data.columns.get_loc(c) for c in target_cols]

    for start in range(len(data) - W - H + 1):

        end = start + W
        pred_idx = end + H - 1

        X.append(data.iloc[start:end].values)
        y.append(data.iloc[pred_idx, target_idx].values)

    return np.array(X), np.array(y)


# =========================================================
# 4️⃣ Flatten Windows
# =========================================================
def flatten_windows(X_train, X_test):

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, X_test_flat


# =========================================================
# 5️⃣ Scaling
# =========================================================
def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# =========================================================
# 6️⃣ Train Model
# =========================================================
def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model training completed!")

    return model


# =========================================================
# 7️⃣ Save Artifacts
# =========================================================
def save_artifacts(model, scaler, model_dir="models"):

    os.makedirs(model_dir, exist_ok=True)

    model_path  = os.path.join(model_dir, "rf_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved at  : {model_path}")
    print(f"Scaler saved at : {scaler_path}")


# =========================================================
# 8️⃣ Main Pipeline
# =========================================================
def main():

    # Paths
    train_path = "data/interim/train_processed.csv"
    test_path  = "data/interim/test_processed.csv"
    
    logging.info("Loading train and test data...")
    # 1️⃣ Load
    train_df, test_df = load_data(train_path, test_path)

    # 2️⃣ Features
    train_df, test_df, target_cols = select_features(
        train_df, test_df
    )

    # 3️⃣ Sliding windows
    W, H = 30, 30
  
    logging.info("Creating sliding windows...") 
    X_train, y_train = create_sliding_windows(
        train_df, target_cols, W, H
    )

    X_test, y_test = create_sliding_windows(
        test_df, target_cols, W, H
    )

    print(f"Sliding windows → Train: {X_train.shape}, Test: {X_test.shape}")

    # 4️⃣ Flatten
    X_train, X_test = flatten_windows(X_train, X_test)

    # 5️⃣ Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    logging.info("Model training...")
    # 6️⃣ Train
    model = train_model(X_train, y_train)
    logging.info("Model training completed!")

    logging.info("Saving artifacts...")
    # 7️⃣ Save
    save_artifacts(model, scaler)
    logging.info("Artifacts saved successfully!")




# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()
