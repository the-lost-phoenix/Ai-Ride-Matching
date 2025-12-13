import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
class DemandLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(DemandLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

class PricingEngine:
    def __init__(self):
        print("⚙️ Initializing Pricing Engine...")
        
        # 1. LOAD ETA MODEL
        self.eta_model = joblib.load("eta_model.pkl")
        
        # 2. LOAD DEMAND MODEL
        self.demand_model = DemandLSTM(hidden_size=64)
        self.demand_model.load_state_dict(torch.load("demand_lstm.pth"))
        self.demand_model.eval()
        
        # Load scaler
        self.scaler = joblib.load("demand_scaler.pkl")
        
        print("✅ Models Loaded Successfully.")

    def predict_eta(self, distance_km, traffic_multiplier, timestamp):
        """
        Uses XGBoost/GradientBoosting to predict trip duration
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        input_data = pd.DataFrame({
            'trip_distance_km': [distance_km],
            'traffic_multiplier': [traffic_multiplier],
            'hour': [hour],
            'day_of_week': [day_of_week]
        })
        
        predicted_minutes = self.eta_model.predict(input_data)[0]
        
        # FIX: Convert numpy float to python float
        return float(max(5, round(predicted_minutes, 1)))

    def predict_demand_multiplier(self, recent_ride_counts):
        """
        Uses LSTM to predict next hour's demand
        """
        inputs_scaled = self.scaler.transform(np.array(recent_ride_counts).reshape(-1, 1))
        inputs_tensor = torch.FloatTensor(inputs_scaled).unsqueeze(0)
        
        with torch.no_grad():
            prediction_scaled = self.demand_model(inputs_tensor)
        
        predicted_demand = self.scaler.inverse_transform(prediction_scaled.numpy())[0][0]
        
        # Logic
        base_multiplier = 1.0
        if predicted_demand > 50:
            base_multiplier = 1.2
        if predicted_demand > 75:
            base_multiplier = 1.5
        if predicted_demand > 100:
            base_multiplier = 2.0
            
        # FIX: Convert to standard Python int and float
        return int(round(predicted_demand, 0)), float(base_multiplier)

    def calculate_price(self, distance_km, duration_min, multiplier):
        """
        Standard Ride Formula
        """
        BASE_FARE = 50
        PER_KM_RATE = 12
        PER_MIN_RATE = 2
        
        price = (BASE_FARE + (distance_km * PER_KM_RATE) + (duration_min * PER_MIN_RATE)) * multiplier
        
        # FIX: Ensure final price is a standard float
        return float(round(price, 2))

# --- TEST CODE ---
if __name__ == "__main__":
    engine = PricingEngine()
    print("\n--- TEST SCENARIO ---")
    mock_time = datetime(2024, 1, 1, 18, 0, 0)
    
    eta = engine.predict_eta(15, 1.5, mock_time)
    print(f"Predicted ETA: {eta} mins (Type: {type(eta)})")
    
    fake_history = [20] * 24
    pred_demand, surge = engine.predict_demand_multiplier(fake_history)
    print(f"Demand: {pred_demand}, Surge: {surge} (Type: {type(pred_demand)})")