import streamlit as st
import requests

# Update this with your FastAPI host if it's deployed on a server
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Temperature Prediction", layout="centered")
st.title("🚗 Battery Temperature Forecast")
st.markdown("Enter a **Vehicle ID (IMEI)** to get the predicted battery temperatures 5 minutes into the future.")

# Input Form
vehicle_id = st.text_input("📟 Vehicle ID (IMEI)", placeholder="e.g., 861557068891727")

if st.button("🔮 Predict Temperature"):
    if not vehicle_id.strip():
        st.warning("Please enter a valid Vehicle ID.")
    else:
        with st.spinner("Fetching live data and predicting..."):
            try:
                response = requests.post(API_URL, json={"vehicle_id": vehicle_id})
                result = response.json()

                if response.status_code == 200 and "prediction" in result:
                    st.success(f"✅ Prediction Successful for Vehicle ID: {vehicle_id}")
                    st.subheader("📊 Predicted Temperatures (After 5 Minutes)")
                    st.markdown(f"🕒 **Timestamp**: `{result['timestamp']}`")

                    for key, value in result["prediction"].items():
                        st.metric(label=key, value=f"{value} °C")
                else:
                    # Show custom message returned from backend (e.g., vehicle is off)
                    error_message = result.get("detail", "Unknown error")
                    st.warning(f"⚠️ {error_message}")

            except Exception as e:
                st.error(f"❌ Failed to connect to FastAPI: {e}")
