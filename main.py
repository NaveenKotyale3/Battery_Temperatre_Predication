from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from datetime import datetime,timedelta
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

    data = response.json()


    packet = data

    

    try:
        # Construct input feature dictionary
        input_dict = {
            "SOC": float(packet.get("soc", 0)),
            "SOH": float(packet.get("soh", 0)),
            "Voltage": float(packet.get("battery_pack_total_voltage", 0)),
            "Current": float(packet.get("battery_current", 0)),
            "Cell Volt 1": float(packet.get("battery_1_volt", 0)),
            "Cell Volt 2": float(packet.get("battery_2_volt", 0)),
            "Cell Volt 3": float(packet.get("battery_3_volt", 0)),
            "Cell Volt 4": float(packet.get("battery_4_volt", 0)),
            "Cell Volt 5": float(packet.get("battery_5_volt", 0)),
            "Cell Volt 6": float(packet.get("battery_6_volt", 0)),
            "Cell Volt 7": float(packet.get("battery_7_volt", 0)),
            "Cell Volt 8": float(packet.get("battery_8_volt", 0)),
            "Cell Volt 9": float(packet.get("battery_9_volt", 0)),
            "Cell Volt 10": float(packet.get("battery_10_volt", 0)),
            "Cell Volt 11": float(packet.get("battery_11_volt", 0)),
            "Cell Volt 12": float(packet.get("battery_12_volt", 0)),
            "Cell Volt 13": float(packet.get("battery_13_volt", 0)),
            "Cell Volt 14": float(packet.get("battery_14_volt", 0)),
            "Cell Volt 15": float(packet.get("battery_15_volt", 0)),
            "Cell Volt 16": float(packet.get("battery_16_volt", 0)),
            "Temperature Sensor 1": float(packet.get("temperature_1", 0)),
            "Temperature Sensor 2": float(packet.get("temperature_2", 0)),
            "Temperature Sensor 3": float(packet.get("temperature_3", 0)),
            "Temperature Sensor 4": float(packet.get("temperature_4", 0)),
            "Temperature Sensor 5": float(packet.get("temperature_5", 0)),
            "Temperature Sensor 6": float(packet.get("temperature_6", 0)),
        }

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

