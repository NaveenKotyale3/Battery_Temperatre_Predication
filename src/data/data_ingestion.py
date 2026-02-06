import os
import requests
import json
import pandas as pd
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging



# =========================================================
# Function: fetch_data_from_api
# =========================================================


def load_params(params_path: str)->dict:
    """
    Load parameters from a Yaml file.

    Args:
        params_path (str): Path to the Yaml file containing parameters.

    Returns:
        dict: Dictionary containing loaded parameters.
    """
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        logging.error(f"Parameters file not found: {params_path}")
        raise

    except yaml.YAMLError as e:
        logging.error(f"Error loading parameters: {e}")
        raise

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}") 
        raise




def fetch_data_from_api(imei: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetches data from the internal API and saves it as a CSV file.

    Args:
        imei (str): IMEI number of the bike/device
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame containing fetched data
    """
    load_dotenv()  # Load environment variables from .env

    # API URL
    api_url = f"https://internalapi.e3techworld.com/bike-unified?imei={imei}&from_date={from_date}&to_date={to_date}"

    # Optional: authentication token (can also be stored in .env)
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        logging.error("API token not found. Please set API_TOKEN in your .env file.")
        raise EnvironmentError("Missing API token")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_token}"
    }


    try:
        logging.info(f"Fetching data from API for IMEI {imei} from {from_date} to {to_date}...")
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            logging.error(f"API request failed with status code {response.status_code}")
            return pd.DataFrame()

        data = response.json()

        # Convert JSON to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.json_normalize(data)
        else:
            raise ValueError("Unexpected JSON structure")

        if df.empty:
            logging.warning("Fetched data is empty.")
        else:
            logging.info(f"Successfully fetched {len(df)} records.")

        # Create Data folder
        # os.makedirs("Data", exist_ok=True)    

        # # Save file dynamically
        # filename = f"Data/data_{imei}_{from_date}_to_{to_date}.csv"
        # df.to_csv(filename, index=False)
        # logging.info(f"Data saved successfully: {filename}")

        return df


    except requests.exceptions.RequestException as e:
        logging.exception(f"Network error while fetching API data: {e}")
        return pd.DataFrame()

    except Exception as e:
        logging.exception(f"Error during data ingestion: {e}")
        return pd.DataFrame()
    


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame,data_path: str)->None:
    """ Save train and test data to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data = train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data = test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.info("Data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise e
    


# =========================================================
# Entry point
# =========================================================
def main():

    try:
        imei = "867512077480024"
        from_date = "2026-02-02"
        to_date = "2026-02-04"

        params = load_params("config/params.yaml")
        test_size = params['data_ingestion']['test_size']

        df = fetch_data_from_api(imei, from_date, to_date)
        if not df.empty:
            logging.info("Data ingestion completed successfully.")
        else:
            logging.warning("Data ingestion completed but no data fetched.")

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, "data")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

    

