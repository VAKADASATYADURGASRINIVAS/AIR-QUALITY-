import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
file_path = "model/city_day_cleaned.csv"  # Ensure dataset is in the correct path
df = pd.read_csv(file_path)

# Selecting features & target
features = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
target = "AQI"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model/model.pkl")
print("✅ Model trained and saved as model.pkl")
