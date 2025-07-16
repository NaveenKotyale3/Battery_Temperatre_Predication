from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

app = FastAPI()

# Load model and feature list
model = joblib.load("temperature_model_no_lag.pkl")
feature_cols = joblib.load("feature_list_no_lag.pkl")

# Zeliot API constants
ZELIOT_BASE_URL = "https://demo.condense.zeliot.in/2251913a-69a6-403a-8561-621de21172cb/latest-packet?uniqueId="
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3R1c2VyIiwiaWF0IjoxNjg1NzM0NjY3LCJleHAiOjE2ODU4MjE4Njd9.9Zw2c2h-D4x7Un2qWlw3G1DcI6x5k7NXbQ6fvVQpVYs"

# Request schema
class VehicleRequest(BaseModel):
    vehicle_id: str  # IMEI number like '861557068891727'

@app.get("/")
def home():
    return {"message": "Live Sensor Temperature Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
def predict_temperature(request: VehicleRequest):
    vehicle_id = request.vehicle_id
    url = f"{ZELIOT_BASE_URL}{vehicle_id}"
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Fetch data from Zeliot API
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from Zeliot API")

    packet = response.json()

    try:
        # Extract input features from the API packet
        input_dict = {
            "SOC": packet.get("soc"),
            "SOH": packet.get("soh"),
            "Voltage": packet.get("battery_pack_total_voltage"),
            "Current": packet.get("battery_current"),
            "Cell Volt 1": packet.get("battery_1_volt"),
            "Cell Volt 2": packet.get("battery_2_volt"),
            "Cell Volt 3": packet.get("battery_3_volt"),
            "Cell Volt 4": packet.get("battery_4_volt"),
            "Cell Volt 5": packet.get("battery_5_volt"),
            "Cell Volt 6": packet.get("battery_6_volt"),
            "Cell Volt 7": packet.get("battery_7_volt"),
            "Cell Volt 8": packet.get("battery_8_volt"),
            "Cell Volt 9": packet.get("battery_9_volt"),
            "Cell Volt 10": packet.get("battery_10_volt"),
            "Cell Volt 11": packet.get("battery_11_volt"),
            "Cell Volt 12": packet.get("battery_12_volt"),
            "Cell Volt 13": packet.get("battery_13_volt"),
            "Cell Volt 14": packet.get("battery_14_volt"),
            "Cell Volt 15": packet.get("battery_15_volt"),
            "Cell Volt 16": packet.get("battery_16_volt"),
            "Temperature Sensor 1": packet.get("temperature_1"),
            "Temperature Sensor 2": packet.get("temperature_2"),
            "Temperature Sensor 3": packet.get("temperature_3"),
            "Temperature Sensor 4": packet.get("temperature_4"),
            "Temperature Sensor 5": packet.get("temperature_5"),
            "Temperature Sensor 6": packet.get("temperature_6"),
        }

        # Check if all values are None or missing
        if all(value is None for value in input_dict.values()):
            return {
                "vehicle_id": vehicle_id,
                "message": "Vehicle appears to be off or data is currently unavailable. Prediction cannot be performed."
            }

        # Check for partial missing values
        if any(value is None for value in input_dict.values()):
            return {
                "vehicle_id": vehicle_id,
                "message": "Some sensor values are missing. Please ensure the vehicle is connected and sending complete data."
            }

        # Convert all values to float
        input_dict = {k: float(v) for k, v in input_dict.items()}

        # Create DataFrame with required columns
        df = pd.DataFrame([input_dict])[feature_cols]

        # Predict
        prediction = model.predict(df)[0]
        prediction = [round(float(p), 2) for p in prediction]

        ist_time = datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(minutes=5)

        return {
            "vehicle_id": vehicle_id,
            "prediction": {
                "BMS Temperature": prediction[0],
                "Battery Pack Temperature 1": prediction[1],
                "Battery Pack Temperature 2": prediction[2],
                "Battery Pack Temperature 3": prediction[3],
                "Battery Pack Temperature 4": prediction[4],
            },
            "timestamp": ist_time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
