import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. LOAD DATA
print("⏳ Loading data...")
df = pd.read_csv("simulated_ride_data.csv")

# 2. FEATURE ENGINEERING (Preparing the data)
# The model can't read "2024-01-01 08:30:00" directly. 
# We must break it down into numbers: "Hour 8", "Monday (0)".
print("🛠️ Processing features...")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# DEFINE INPUTS (X) AND OUTPUT (y)
# We use these columns to predict the duration
features = ['trip_distance_km', 'traffic_multiplier', 'hour', 'day_of_week']
target = 'trip_duration_min'

X = df[features]
y = df[target]

# 3. SPLIT DATA
# We keep 20% of data hidden to test the model later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL
# GradientBoosting is powerful for this type of regression task
print("🤖 Training the ETA Model (this might take a few seconds)...")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
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