# =========================================================
# Imports
# =========================================================
import os
import json
import joblib
import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from src.model.model_building import create_sliding_windows

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.logger import logging


# =========================================================
# MLflow + DagsHub Initialization
# =========================================================
mlflow.set_tracking_uri('https://dagshub.com/NaveenKotyale3/Battery_Temperatre_Predication.mlflow')
dagshub.init(repo_owner='NaveenKotyale3', repo_name='Battery_Temperatre_Predication', mlflow=True)


# =========================================================
# Load Model
# =========================================================
def load_model(model_path: str):

    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model

    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


# =========================================================
# Load Scaler
# =========================================================
def load_scaler(scaler_path: str):

    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Scaler loaded from {scaler_path}")
        return scaler

    except FileNotFoundError:
        logging.error(f"Scaler file not found: {scaler_path}")
        raise

    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise


# =========================================================
# Load Data
# =========================================================
def load_data(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path} | Shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# =========================================================
# Target Columns
# =========================================================
def get_target_columns():

    return [f"temperature_{i}" for i in range(1, 7) if i != 2]


# =========================================================
# Feature Selection
# =========================================================
def select_features(df: pd.DataFrame):

    input_columns = (
        ["SOC", "SOH", "Battery_pack_total_voltage", "Battery_current"]
        + [f"Battery_{i}_Volt" for i in range(1, 17)]
        + [f"temperature_{i}" for i in range(1, 7)]
    )

    df = df[input_columns].dropna().reset_index(drop=True)

    logging.info(f"Feature-selected data shape: {df.shape}")

    return df


# =========================================================
# Sliding Window Creation
# =========================================================


# =========================================================
# Flatten Windows
# =========================================================
def flatten_windows(X):

    X_flat = X.reshape(X.shape[0], -1)

    logging.info(f"Flattened shape: {X_flat.shape}")

    return X_flat


# =========================================================
# Evaluate Model
# =========================================================
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        "r2_score": r2,
        "mean_absolute_error": mae,
        "root_mean_squared_error": rmse
    }

    logging.info(f"Evaluation metrics: {metrics}")

    return metrics


# =========================================================
# Save Metrics
# =========================================================
def save_metrics(metrics: dict, file_path: str):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics saved to {file_path}")


# =========================================================
# Save Run Info
# =========================================================
def save_model_info(run_id: str, model_path: str, file_path: str):

    info = {
        "run_id": run_id,
        "model_path": model_path
    }

    with open(file_path, "w") as f:
        json.dump(info, f, indent=4)

    logging.info(f"Run info saved to {file_path}")


# =========================================================
# MAIN EVALUATION PIPELINE
# =========================================================
def main():

    try:

        mlflow.set_experiment("Battery Temperature Prediction")

        with mlflow.start_run() as run:

            logging.info("MLflow run started")

            # -----------------------------------
            # Load artifacts
            # -----------------------------------
            model  = load_model("models/rf_model.pkl")
            scaler = load_scaler("models/scaler.pkl")

            # -----------------------------------
            # Load test data
            # -----------------------------------
            test_df = load_data(
                "data/interim/test_processed.csv"
            )

            # -----------------------------------
            # Feature selection
            # -----------------------------------
            test_df = select_features(test_df)

            target_cols = get_target_columns()

            # -----------------------------------
            # Sliding windows
            # -----------------------------------
            X_test, y_test = create_sliding_windows(
                test_df,
                target_cols,
                W=30,
                H=30
            )

            # -----------------------------------
            # Flatten + Scale
            # -----------------------------------
            X_test = flatten_windows(X_test)
            X_test = scaler.transform(X_test)

            # -----------------------------------
            # Evaluate
            # -----------------------------------
            metrics = evaluate_model(model, X_test, y_test)

            # -----------------------------------
            # Save metrics
            # -----------------------------------
            metrics_path = "reports/metrics.json"
            save_metrics(metrics, metrics_path)

            # -----------------------------------
            # Log MLflow metrics
            # -----------------------------------
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log params
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # Log model and capture the model info
            model_info = mlflow.sklearn.log_model(model, "model")

            # Log artifacts
            mlflow.log_artifact(metrics_path)

            # Save run info with model_uri for proper registration
            save_model_info(
                run.info.run_id,
                model_info.model_uri,
                "reports/experiment_info.json"
            )

            logging.info("Evaluation pipeline completed successfully")

    except Exception as e:
        logging.exception(f"Evaluation pipeline failed: {e}")
        raise


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()
