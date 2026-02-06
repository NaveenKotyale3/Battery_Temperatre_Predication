import pandas as pd
import numpy as np
from src.logger import logging
import os


# =========================================================
# Data Preprocessing Function
# =========================================================
def preprocess_data(df):
    """
    Reads raw CSV data, performs cleaning & filtering, and saves processed output.
    """

    # logging.info(f"Reading input CSV file: {input_path}")
    # df = pd.read_csv(input_path)
    logging.info(f"Initial rows: {len(df)}")

    logging.info("Selecting relevant features...")
    features = [
        "timestamp", "SOH", "Cycle_Count", "SOC",
        "Battery_1_Volt", "Battery_2_Volt", "Battery_3_Volt", "Battery_4_Volt",
        "Battery_5_Volt", "Battery_6_Volt", "Battery_7_Volt", "Battery_8_Volt",
        "Battery_9_Volt", "Battery_10_Volt", "Battery_11_Volt", "Battery_12_Volt",
        "Battery_13_Volt", "Battery_14_Volt", "Battery_15_Volt", "Battery_16_Volt",
        "temperature_1", "temperature_2", "temperature_3", "temperature_4",
        "temperature_5", "temperature_6",
        "Battery_current", "Residual_battery_energy", "Remaining_Capacity",
        "Full_Capacity", "Full_battery_energy", "error_code", "Battery_pack_total_voltage"
    ]

    df = df[features]

    logging.info("Filtering valid timestamps...")
    # Drop invalid timestamps
    df = df.dropna(subset=["timestamp"])

    df["timestamp"] = pd.to_datetime(df["timestamp"],errors="coerce",utc=True)

    # 2️ Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute time difference (in seconds)
    df["time_diff_sec"] = df["timestamp"].diff().dt.total_seconds()

    # Keep only rows with time gap in [5s, 15s]
    df = df[
        (df["time_diff_sec"].between(5, 15)) |
        (df["time_diff_sec"].isna())   # keep first row
    ]

    #imputing the missing values 

    # Temperature
    temp_cols = [f"temperature_{i}" for i in range(1, 7)]
    df[temp_cols] = df[temp_cols].apply(
        lambda row: row.fillna(row.mean()), axis=1
    )

    # Voltages
    volt_cols = [c for c in df.columns if c.startswith("Battery_") and c.endswith("_Volt")]
    df[volt_cols] = df[volt_cols].apply(
        lambda row: row.fillna(row.mean()), axis=1
    )

    #conversion in into mV to V
    df[volt_cols] = df[volt_cols] / 1000.0

    # Pack voltage
    df["Battery_pack_total_voltage"] = df[volt_cols].sum(axis=1)

    # Stateful signals
    df[["SOC", "SOH", "Cycle_Count"]] = df[["SOC", "SOH", "Cycle_Count"]].ffill()

    # Capacity
    df[["Remaining_Capacity", "Full_Capacity"]] = df[["Remaining_Capacity", "Full_Capacity"]].ffill()

    logging.info("Resetting index...")
    df.reset_index(drop=True, inplace=True)

    logging.info("Dropping the null values ")
    df.dropna(inplace=True)


    logging.info(f"Final cleaned rows: {len(df)}")

    return df


def main():
    try:
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logging.info("Data loaded successfully.")

        train_processed_data = preprocess_data(train_data)
        test_processed_data = preprocess_data(test_data)
        logging.info("Data preprocessing completed.")

        clean_data_path =os.path.join("data", "interim")
        os.makedirs(clean_data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(clean_data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(clean_data_path, "test_processed.csv"), index=False)
        logging.info("Processed data saved successfully.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()