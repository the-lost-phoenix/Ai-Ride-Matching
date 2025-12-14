import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# --- CONFIGURATION ---
NUM_ROWS = 20000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "simulated_ride_data.csv")

# Real Bangalore Hotspots (Lat, Lon) for realism
LOCATIONS = {
    "Koramangala": (12.9352, 77.6245),
    "Indiranagar": (12.9784, 77.6408),
    "Whitefield": (12.9698, 77.7500),
    "MG Road": (12.9716, 77.5946),
    "Electronic City": (12.8452, 77.6602),
    "Jayanagar": (12.9308, 77.5838),
    "Malleshwaram": (13.0031, 77.5643),
    "HSR Layout": (12.9121, 77.6446),
    "Marathahalli": (12.9592, 77.6974),
    "Kempegowda Int'l Airport": (13.1986, 77.7066),
    "Hebbal": (13.0359, 77.5970),
    "Banashankari": (12.9255, 77.5468),
    "Domlur": (12.9609, 77.6387),
    "Rajajinagar": (12.9915, 77.5561),
    "BTM Layout": (12.9166, 77.6101)
}

VEHICLES = ["Bike", "Auto", "Mini", "Prime", "SUV"]

def generate_synthetic_data():
    print(f"Generating {NUM_ROWS} synthetic rides (Instant mode)...")
    
    data = []
    location_names = list(LOCATIONS.keys())
    
    # Vectorized generation would be faster, but loop is clearer for logic customization
    # For 20k rows, python loop is fine (<1 sec)
    
    start_date = datetime(2024, 1, 1)
    
    for _ in range(NUM_ROWS):
        # Pick Origin and Dest
        origin_name = random.choice(location_names)
        dest_name = random.choice(location_names)
        while origin_name == dest_name:
            dest_name = random.choice(location_names)
            
        orig_lat, orig_lon = LOCATIONS[origin_name]
        dest_lat, dest_lon = LOCATIONS[dest_name]
        
        # Add slight random noise to coords so they aren't all identical points
        orig_lat += random.uniform(-0.01, 0.01)
        orig_lon += random.uniform(-0.01, 0.01)
        dest_lat += random.uniform(-0.01, 0.01)
        dest_lon += random.uniform(-0.01, 0.01)
        
        # Calculate straight line distance (Haversine approx)
        # Using simple euclidean approx for speed since it's local: 1 deg lat ~ 111km
        lat_diff = (dest_lat - orig_lat) * 111
        lon_diff = (dest_lon - orig_lon) * 111 * 0.97 # cos(lat) adjustment
        distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 1.3 # 1.3x for road curvature
        
        # Time and Traffic
        random_days = random.randint(0, 60)
        random_seconds = random.randint(0, 86400)
        timestamp = start_date + timedelta(days=random_days, seconds=random_seconds)
        hour = timestamp.hour
        
        # Weather Logic (New Feature)
        # Random weather, but weighted (Bangalore is mostly pleasant/cloudy, sometimes rain)
        weather_conditions = ["Sunny", "Cloudy", "Rainy"]
        weather = random.choices(weather_conditions, weights=[60, 30, 10])[0]
        
        # Traffic Multiplier
        traffic_multiplier = 1.0
        traffic_label = "Low"
        
        if 8 <= hour <= 11 or 17 <= hour <= 21:
            traffic_multiplier = 1.8 # Peak
            traffic_label = "High"
        elif 23 <= hour or hour <= 5:
            traffic_multiplier = 0.9 # Night
            traffic_label = "Free Flow"
        else:
            traffic_multiplier = 1.3 # Normal
            traffic_label = "Medium"
            
        # Weather Impact on Traffic
        if weather == "Rainy":
            traffic_multiplier *= 1.2
            traffic_label = "Heavy Rain"
            
        duration_min = (distance_km * 3.0) * traffic_multiplier 
        
        # Price
        vehicle = random.choice(VEHICLES)
        base_price = 50
        max_passengers = 4
        if vehicle == "Bike": 
            base_price = 30
            max_passengers = 1
        elif vehicle == "Auto": 
            base_price = 40
            max_passengers = 3
        elif vehicle == "SUV": 
            base_price = 100
            max_passengers = 6
        elif vehicle == "Mini":
            max_passengers = 4
            
        passenger_count = random.randint(1, max_passengers)
        driver_rating = round(random.uniform(3.5, 5.0), 1)
        
        fare = base_price + (distance_km * 12) + (duration_min * 1.5)
        if weather == "Rainy":
            fare *= 1.1 # Rain surge
        
        data.append({
            "origin_lat": round(orig_lat, 6),
            "origin_lon": round(orig_lon, 6),
            "dest_lat": round(dest_lat, 6),
            "dest_lon": round(dest_lon, 6),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_type": vehicle,
            "trip_distance": round(distance_km, 2),
            "trip_duration": round(duration_min, 1),
            "fare": round(fare, 2),
            # --- NEW FEATURES ---
            "traffic_condition": traffic_label,
            "weather": weather,
            "passenger_count": passenger_count,
            "driver_rating": driver_rating,
            "surge_multiplier": round(traffic_multiplier, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df)} rows in '{OUTPUT_FILE}'")

if __name__ == "__main__":
    generate_synthetic_data()
