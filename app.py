from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")

# AQI Classification Table
AQI_CATEGORIES = [
    (0, 50, "Good", "✅ Excellent", "No health impacts"),
    (51, 100, "Satisfactory", "✅ Acceptable", "Minor breathing issues for sensitive individuals"),
    (101, 200, "Moderate", "⚠️ Slightly Polluted", "May cause discomfort for sensitive groups"),
    (201, 300, "Poor", "❌ Unhealthy", "Breathing discomfort for people with asthma, elderly, and children"),
    (301, 400, "Very Poor", "❌ Hazardous", "Increased respiratory issues, potential long-term effects"),
    (401, 500, "Severe", "☠️ Toxic Air", "Serious health risks for everyone"),
]

def get_aqi_info(aqi_value):
    """Returns AQI category, air quality, and health effects based on AQI value."""
    for min_val, max_val, category, quality, effects in AQI_CATEGORIES:
        if min_val <= aqi_value <= max_val:
            return category, quality, effects
    return "Unknown", "Unknown", "Unknown"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Extract input values
    features = [data["pm25"], data["pm10"], data["co"], data["no2"], data["so2"], data["o3"]]
    
    # Predict AQI
    predicted_aqi = model.predict([features])[0]
    
    # Get AQI details
    category, air_quality, health_effects = get_aqi_info(predicted_aqi)
    
    return jsonify({
        "aqi": round(predicted_aqi, 2),
        "category": category,
        "air_quality": air_quality,
        "health_effects": health_effects
    })

if __name__ == "__main__":
    app.run(debug=True)
