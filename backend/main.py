from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORT THIS
from pydantic import BaseModel
from pricing_engine import PricingEngine
from datetime import datetime
import random

app = FastAPI(title="AI Ride Matching System")

# --- ENABLE CORS (Allow React to talk to Python) ---
origins = [
    "http://localhost:5173",  # React Localhost
    "http://127.0.0.1:5173",
    "*"                       # Allow all (for development)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting API Server...")
engine = PricingEngine()# This loads your AI models instantly

# 2. DEFINE REQUEST FORMAT (Data Validation)
# The API expects this exact data from the user
class RideRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    user_preference: str = "balanced" # options: "cheapest", "fastest", "balanced"

class VehicleUpdate(BaseModel):
    vehicle_id: str
    lat: float
    lon: float
    status: str # "available", "busy"

# 3. HELPER FUNCTION: DISTANCE CALC
# Simple Haversine approx or Euclidean for speed
def calculate_distance(lat1, lon1, lat2, lon2):
    # Approx: 1 deg lat = 111km
    d_lat = (lat2 - lat1) * 111
    d_lon = (lon2 - lon1) * 111
    return (d_lat**2 + d_lon**2)**0.5

# 4. API ENDPOINT: GET RIDE QUOTE
# This matches the PDF requirement: POST /ride/quote [cite: 23]
@app.post("/ride/quote")
def get_ride_quotes(request: RideRequest):
    
    # A. Calculate Trip Details
    distance_km = calculate_distance(request.origin_lat, request.origin_lon, 
                                     request.dest_lat, request.dest_lon)
    
    # Simulate real-time context (In a real app, this comes from live sensors)
    current_time = datetime.now()
    traffic_multiplier = 1.5 if (8 <= current_time.hour <= 10) else 1.1
    
    # B. Ask the AI Brain
    # 1. Get ETA
    eta_min = engine.predict_eta(distance_km, traffic_multiplier, current_time)
    
    # 2. Get Surge Pricing (Using fake history for demo)
    # In production, this would query a database for actual last 24h ride counts
    fake_history = [random.randint(20, 100) for _ in range(24)]
    predicted_demand, surge_multiplier = engine.predict_demand_multiplier(fake_history)
    
    # C. Generate Vehicle Options (Vehicle Ranking Logic)
    # We create 3 virtual cars to demonstrate ranking
    
    # Option 1: Standard (The AI prediction)
    price_std = engine.calculate_price(distance_km, eta_min, surge_multiplier)
    
    # Option 2: Premium (Faster but expensive)
    eta_prem = eta_min * 0.8 # 20% faster
    price_prem = price_std * 1.5
    
    # Option 3: Eco (Slower but cheaper)
    eta_eco = eta_min * 1.2 # 20% slower
    price_eco = price_std * 0.7
    
    options = [
        {"type": "Standard", "price": round(price_std, 2), "eta": round(eta_min, 1), "surge": surge_multiplier},
        {"type": "Premium", "price": round(price_prem, 2), "eta": round(eta_prem, 1), "surge": surge_multiplier},
        {"type": "Eco_Saver", "price": round(price_eco, 2), "eta": round(eta_eco, 1), "surge": surge_multiplier}
    ]
    
    # D. Rank/Sort based on User Preference 
    if request.user_preference == "cheapest":
        options.sort(key=lambda x: x['price'])
    elif request.user_preference == "fastest":
        options.sort(key=lambda x: x['eta'])
    # "balanced" keeps default order
    
    return {
        "trip_distance_km": round(distance_km, 2),
        "predicted_demand_next_hour": predicted_demand,
        "surge_applied": f"{surge_multiplier}x",
        "options": options
    }

# 5. API ENDPOINT: UPDATE VEHICLE
# Matches PDF requirement: POST /vehicles/update [cite: 22]
@app.post("/vehicles/update")
def update_vehicle(update: VehicleUpdate):
    # In a real app, save this to DB. Here we just acknowledge.
    return {"status": "success", "msg": f"Vehicle {update.vehicle_id} location updated."}

# To run: uvicorn main:app --reload