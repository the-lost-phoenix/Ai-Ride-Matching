import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- CONFIGURATION ---
NUM_ROWS = 10000  # Number of trips to simulate
START_DATE = datetime(2024, 1, 1)

# Define a "Fake City" (Approximate Lat/Lon for Bangalore, India)
CITY_LAT = 12.9716
CITY_LON = 77.5946
RADIUS = 0.05  # roughly 5-6km radius

print("🚀 Starting Data Generation...")

# 1. GENERATE TIMESTAMPS (With realistic patterns)
# We want more rides during "Peak Hours" (8am-10am and 5pm-8pm)
timestamps = []
current_time = START_DATE

for _ in range(NUM_ROWS):
    # Advance time: Randomly add 1 to 5 minutes between requests
    # This creates a sequential timeline
    current_time += timedelta(minutes=random.randint(1, 5))
    timestamps.append(current_time)

# 2. GENERATE LOCATIONS & FEATURES
data = []

for t in timestamps:
    # --- SIMULATE TIME-BASED TRAFFIC ---
    hour = t.hour
    is_weekend = t.weekday() >= 5
    
    # Simple logic: Rush hour is 8-10 AM and 5-8 PM on weekdays
    is_rush_hour = (8 <= hour <= 10) or (17 <= hour <= 20)
    if is_weekend:
        is_rush_hour = False # Less traffic on weekends

    # --- TRAFFIC FACTOR ---
    # 1.0 = Empty roads, 2.0 = Heavy Traffic
    traffic_multiplier = 1.0
    if is_rush_hour:
        traffic_multiplier = random.uniform(1.5, 2.5) # Heavy traffic
    else:
        traffic_multiplier = random.uniform(1.0, 1.3) # Normal traffic

    # --- LOCATIONS ---
    # Random points around the city center
    origin_lat = CITY_LAT + random.uniform(-RADIUS, RADIUS)
    origin_lon = CITY_LON + random.uniform(-RADIUS, RADIUS)
    dest_lat = CITY_LAT + random.uniform(-RADIUS, RADIUS)
    dest_lon = CITY_LON + random.uniform(-RADIUS, RADIUS)
    
    # Calculate Straight-line Distance (Euclidean approximation for simplicity)
    # Degree to KM conversion approx: 111km per degree
    dist_lat = (dest_lat - origin_lat) * 111
    dist_lon = (dest_lon - origin_lon) * 111
    distance_km = np.sqrt(dist_lat**2 + dist_lon**2)

    # --- CALCULATE DURATION (THE TARGET VARIABLE) ---
    # Base speed: 30 km/h. 
    # Duration = (Distance / Speed) * Traffic_Multiplier
    # Add some random noise because real world isn't perfect
    base_speed_kmh = 30
    duration_hours = (distance_km / base_speed_kmh) * traffic_multiplier
    duration_mins = duration_hours * 60 + random.uniform(-2, 5) # noise
    
    # Ensure no negative duration
    duration_mins = max(5, duration_mins) 

    # --- PRICING (Ground Truth) ---
    # Base fare $5 + $2 per km + Traffic Surge
    base_fare = 50 
    price = base_fare + (distance_km * 12) * traffic_multiplier

    data.append({
        "timestamp": t,
        "origin_lat": round(origin_lat, 6),
        "origin_lon": round(origin_lon, 6),
        "dest_lat": round(dest_lat, 6),
        "dest_lon": round(dest_lon, 6),
        "trip_distance_km": round(distance_km, 2),
        "traffic_multiplier": round(traffic_multiplier, 2),
        "trip_duration_min": round(duration_mins, 2), # TARGET 1 (ETA)
        "trip_price": round(price, 2)
    })

# 3. SAVE TO CSV
df = pd.DataFrame(data)
df.to_csv("simulated_ride_data.csv", index=False)

print(f"✅ SUCCESS! Generated {NUM_ROWS} rides.")
print(df.head())