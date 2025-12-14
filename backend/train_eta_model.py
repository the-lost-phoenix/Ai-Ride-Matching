import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. LOAD DATA
print("⏳ Loading data...")
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "simulated_ride_data.csv"))

# 2. FEATURE ENGINEERING (Preparing the data)
print("🛠️ Processing features (Weather, Traffic, etc.)...")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Normalize column names to match what Pricing Engine expects
df.rename(columns={'trip_distance': 'trip_distance_km', 'trip_duration': 'trip_duration_min'}, inplace=True)

# ENCODING CATEGORICAL FEATURES
# traffic_condition (Ordinal can be better than one-hot for "intensity")
traffic_map = {"Free Flow": 0, "Low": 0, "Medium": 1, "High": 2, "Heavy Rain": 3}
if 'traffic_condition' in df.columns:
    df['traffic_level'] = df['traffic_condition'].map(traffic_map).fillna(1)
else:
    df['traffic_level'] = 1 # Default

# weather (One-Hot Encoding)
expected_weather_cols = ['weather_Sunny', 'weather_Cloudy', 'weather_Rainy']

if 'weather' in df.columns:
    df = pd.get_dummies(df, columns=['weather'], prefix='weather', drop_first=False)
    for col in expected_weather_cols:
        if col not in df.columns:
            df[col] = 0
else:
    # If using old data without weather, default to sunny
    df['weather_Sunny'] = 1
    df['weather_Cloudy'] = 0
    df['weather_Rainy'] = 0

# DEFINE INPUTS (X) AND OUTPUT (y)
features = [
    'trip_distance_km', 'hour', 'day_of_week', 
    'traffic_level', 
    'weather_Sunny', 'weather_Cloudy', 'weather_Rainy'
]
# Add passenger_count if exists
if 'passenger_count' in df.columns:
    features.append('passenger_count')

target = 'trip_duration_min'

X = df[features]
y = df[target]

# 3. SPLIT DATA
# We keep 20% of data hidden to test the model later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL
# GradientBoosting is powerful for this type of regression task
print("🤖 Training the High-Accuracy ETA Model...")
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATE THE MODEL
# We ask the model to predict the test set and compare with reality
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("-" * 30)
print(f"✅ Model Training Complete!")
print(f"📊 Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"   (On average, our prediction is off by {mae:.2f} mins)")
print(f"📊 Root Mean Squared Error (RMSE): {rmse:.2f}")
print("-" * 30)

# 6. SAVE THE MODEL
# We save the "brain" to a file so the API can use it later
joblib.dump(model, "eta_model.pkl")
print("💾 Model saved as 'eta_model.pkl'")