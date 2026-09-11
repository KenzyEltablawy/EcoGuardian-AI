
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import joblib
import pandas as pd

st.set_page_config(page_title="EcoGuardian AI", page_icon="🌍", layout="wide")
st.title("🌍 EcoGuardian AI")
st.subheader("AI Decision Support System for Climate Intervention Planning")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
city = st.text_input("🌍 Enter a City Name", "Cairo")

if city:
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    ).json()
    if "results" not in geo:
        st.error("City not found.")
        st.stop()

    r = geo["results"][0]
    lat, lon = r["latitude"], r["longitude"]
    country = r.get("country", "")

    w = requests.get(
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    ).json()["current"]

    a = requests.get(
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&current=pm10,pm2_5"
    ).json()["current"]

    temp, hum, wind = w["temperature_2m"], w["relative_humidity_2m"], w["wind_speed_10m"]
    pm25, pm10 = a["pm2_5"], a["pm10"]

    heat_risk = 0.5*temp + 0.3*(100-hum) + 0.2*wind
    pollution_risk = 0.6*pm25 + 0.4*pm10
    severity = 0.6*heat_risk + 0.4*pollution_risk

    st.success(f"📍 {city}, {country}")
    st.markdown("---")
    st.header("🌦 Current Environmental Conditions")
    c1,c2,c3 = st.columns(3)
    c1.metric("🌡 Temperature", f"{temp} °C")
    c2.metric("💧 Humidity", f"{hum} %")
    c3.metric("🌬 Wind", f"{wind} km/h")
    c4,c5 = st.columns(2)
    c4.metric("PM2.5", pm25)
    c5.metric("PM10", pm10)

    st.markdown("---")
    m = folium.Map(location=[lat, lon], zoom_start=11)
    folium.Marker([lat, lon], tooltip=city).add_to(m)
    st_folium(m, width=900, height=400)

    st.markdown("---")
    st.header("🤖 AI Recommendation")
    features = pd.DataFrame([{
        "Temperature": temp, "Humidity": hum, "WindSpeed": wind,
        "PM2.5": pm25, "PM10": pm10,
        "HeatRisk": heat_risk, "PollutionRisk": pollution_risk,
        "ClimateSeverity": severity
    }])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features).max()*100

    st.success(f"### {pred}")
    st.metric("Confidence", f"{proba:.1f}%")

    st.subheader("💡 Why?")
    reasons = []
    if temp > 35: reasons.append("High temperature")
    if hum < 30: reasons.append("Low humidity")
    if pm25 > 35: reasons.append("High PM2.5")
    if pm10 > 50: reasons.append("High PM10")
    if not reasons:
        st.write("✅ No major threshold-based environmental risk detected.")
    else:
        for r_ in reasons:
            st.write(f"• {r_}")

    st.markdown("---")
    st.subheader("📍 Location Information")
    st.write(f"City: {city} | Country: {country}")
    st.write(f"Latitude: {lat} | Longitude: {lon}")
