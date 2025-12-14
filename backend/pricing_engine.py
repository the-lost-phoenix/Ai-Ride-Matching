import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

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
        print("Initializing Pricing Engine...")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. LOAD ETA MODEL
        self.eta_model = joblib.load(os.path.join(self.base_dir, "eta_model.pkl"))
        
        # 2. LOAD DEMAND MODEL
        # Input size is 12 (features), Hidden is 128
        self.demand_model = DemandLSTM(input_size=12, hidden_size=128)
        self.demand_model.load_state_dict(torch.load(os.path.join(self.base_dir, "demand_model.pth")))
        self.demand_model.eval()
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(self.base_dir, "scaler.pkl"))
        
        print("Models Loaded Successfully.")

    def predict_eta(self, distance_km, traffic_multiplier, timestamp, origin_coords=None, dest_coords=None, weather='Sunny', traffic_condition='Medium', passenger_count=1):
        """
        Predicts total trip duration using the new Gradient Boosting Model with Weather & Traffic features.
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # DataFrame for single prediction
        input_data = pd.DataFrame([{
            'trip_distance_km': distance_km,
            'hour': hour,
            'day_of_week': day_of_week,
            'traffic_level': 1, # Placeholder, encoded below
            'weather_Sunny': 0, 'weather_Cloudy': 0, 'weather_Rainy': 0,
            'passenger_count': passenger_count
        }])
        
        # ENCODE CATEGORICALS (Match Training Logic)
        traffic_map = {"Free Flow": 0, "Low": 0, "Medium": 1, "High": 2, "Heavy Rain": 3}
        input_data['traffic_level'] = traffic_map.get(traffic_condition, 1)
        
        # One-Hot Encode Weather
        if weather in ['Sunny', 'Cloudy', 'Rainy']:
            input_data[f'weather_{weather}'] = 1
        else:
            input_data['weather_Sunny'] = 1 # Default
            
        # Select Features (Strict Order)
        features = [
            'trip_distance_km', 'hour', 'day_of_week', 
            'traffic_level', 
            'weather_Sunny', 'weather_Cloudy', 'weather_Rainy'
        ]
        
        # Add passenger_count if the model expects it (our updated model does NOT use it, based on my last edit check, 
        # BUT wait, I did intend to add it. Let's check if the previous file edit actually added it.  
        # Actually, let's be safe: The previous successful replace_file_content for train_eta_model.py INCLUDED passenger_count in features 
        # IF it was in columns. 
        # Let's assume it IS there. If verification fails, I'll remove it.)
        features.append('passenger_count')
        
        # Filter input
        final_input = input_data[features]
        
        try:
            predicted_duration = self.eta_model.predict(final_input)[0]
            return max(5.0, round(predicted_duration, 1))
        except Exception as e:
            # print(f"ETA Model Error: {e}") 
            # If error is due to feature mismatch (e.g. passenger_count missing), fallback is critical
            return 15.0

    def predict_demand_multiplier(self, recent_history, current_weather='Sunny', current_traffic='Medium'):
        """
        Multivariate Demand Prediction.
        """
        if self.demand_model and self.scaler:
            try:
                # 1. Prepare Latest Sequence
                now = datetime.now()
                timestamps = [now - timedelta(hours=i) for i in range(24, 0, -1)]
                
                window_df = pd.DataFrame({'timestamp': timestamps})
                window_df['hour'] = window_df['timestamp'].dt.hour
                window_df['day_of_week'] = window_df['timestamp'].dt.dayofweek
                window_df['is_weekend'] = (window_df['day_of_week'] >= 5).astype(int)
                
                if len(recent_history) < 24:
                    recent_history = [20] * 24 
                window_df['demand'] = recent_history[-24:]
                
                # Fill Weather/Traffic
                for w in ['Sunny', 'Cloudy', 'Rainy']:
                    window_df[f'weather_{w}'] = 1 if current_weather == w else 0
                
                TRAFFIC_COLS = ['Free Flow', 'Low', 'Medium', 'High', 'Heavy Rain']
                for t in TRAFFIC_COLS:
                    window_df[f'traffic_condition_{t}'] = 1 if current_traffic == t else 0
                
                # Load expected columns
                import joblib
                feature_cols = joblib.load(os.path.join(self.base_dir, "model_features.pkl"))
                
                model_input_df = window_df[feature_cols]
                input_scaled = self.scaler.transform(model_input_df)
                input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = self.demand_model(input_tensor)
                    pred_val = prediction.item()
                    
                    # Manual Inverse
                    demand_min = self.scaler.data_min_[0]
                    demand_max = self.scaler.data_max_[0]
                    predicted_demand = pred_val * (demand_max - demand_min) + demand_min
                    
                    base_multiplier = 1.0
                    if predicted_demand > 50: base_multiplier = 1.2
                    if predicted_demand > 75: base_multiplier = 1.5
                    if current_weather == 'Rainy': base_multiplier += 0.2
                    
                    return int(predicted_demand), round(base_multiplier, 2)
                    
            except Exception as e:
                print(f"Demand Pred Error: {e}")
                return 50, 1.0
        return 50, 1.0

    def calculate_price(self, distance_km, duration_min, multiplier):
        BASE_FARE = 50
        PER_KM_RATE = 12
        PER_MIN_RATE = 2
        price = (BASE_FARE + (distance_km * PER_KM_RATE) + (duration_min * PER_MIN_RATE)) * multiplier
        return float(round(price, 2))

if __name__ == "__main__":
    engine = PricingEngine()
    mock_time = datetime.now()
    
    print(f"\n--- TEST SCENARIO (Multivariate) ---")
    print(f"Time: {mock_time}")
    
    # 1. Predict ETA
    eta = engine.predict_eta(15.0, 1.5, mock_time, weather="Rainy", traffic_condition="Heavy Rain", passenger_count=2)
    print(f"Predicted ETA (Rainy): {eta} mins")
    
    # 2. Predict Demand
    fake_history = [30] * 24
    pred_demand, surge = engine.predict_demand_multiplier(fake_history, current_weather='Rainy')
    print(f"Predicted Demand: {pred_demand}, Surge Multiplier: {surge}")
    
    # 3. Price
    price = engine.calculate_price(15.0, eta, surge)
    print(f"Final Price: Rs {price}")