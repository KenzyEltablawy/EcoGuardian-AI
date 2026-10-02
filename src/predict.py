"""
Prediction module for EcoGuardian AI.
Loads trained model and generates intervention recommendations.
"""

import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')


def load_model():
    """Load trained Random Forest model."""
    return joblib.load(MODEL_PATH)


def predict_recommendation(temperature, humidity, wind_speed, pm25, pm10):
    """
    Generate an intervention recommendation with confidence score.
    
    Args:
        temperature (float): Temperature in Celsius
        humidity (float): Relative humidity in percentage
        wind_speed (float): Wind speed in km/h
        pm25 (float): PM2.5 concentration in µg/m³
        pm10 (float): PM10 concentration in µg/m³
    
    Returns:
        dict: recommendation and confidence
    """
    model = load_model()
    
    # Feature Engineering
    heat_risk = (0.5 * temperature) + (0.3 * (100 - humidity)) + (0.2 * wind_speed)
    pollution_risk = (0.6 * pm25) + (0.4 * pm10)
    climate_severity = (0.6 * heat_risk) + (0.4 * pollution_risk)
    
    features = pd.DataFrame([{
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pm25': pm25,
        'pm10': pm10,
        'Heat_Risk': heat_risk,
        'Pollution_Risk': pollution_risk,
        'Climate_Severity': climate_severity
    }])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities) * 100)
    
    return {
        'recommendation': prediction,
        'confidence': round(confidence, 2)
    }


def explain_prediction(temperature, humidity, wind_speed, pm25, pm10):
    """
    Generate threshold-based explanation for the prediction.
    
    Returns:
        list: explanation strings
    """
    explanations = []
    
    if temperature > 30:
        explanations.append("🌡️ Elevated temperature indicates increased urban heat risk.")
    if humidity < 40:
        explanations.append("💧 Low humidity contributes to atmospheric dryness.")
    if wind_speed > 20:
        explanations.append("💨 High wind speed may disperse pollutants.")
    if pm25 > 25:
        explanations.append("🏭 Elevated PM2.5 indicates fine particulate pollution.")
    if pm10 > 50:
        explanations.append("🏭 Elevated PM10 indicates coarse particulate pollution.")
    
    if not explanations:
        explanations.append("✅ No major threshold-based environmental risk detected.")
    
    return explanations


if __name__ == "__main__":
    result = predict_recommendation(30.0, 57.0, 11.7, 16.6, 26.5)
    print(f"🎯 Recommendation: {result['recommendation']}")
    print(f"📊 Confidence: {result['confidence']}%")
    print("\n💡 Explanation:")
    for exp in explain_prediction(30.0, 57.0, 11.7, 16.6, 26.5):
        print(f"  - {exp}")
