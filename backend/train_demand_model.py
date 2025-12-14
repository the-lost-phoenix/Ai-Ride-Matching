import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- CONFIGURATION (UPDATED for "Perfect Accuracy") ---
LOOKBACK_WINDOW = 24  # Look back 24 hours
HIDDEN_SIZE = 128     # Increased complexity
EPOCHS = 50           # More training
LEARNING_RATE = 0.001

print("⏳ Loading data for Advanced Demand Forecasting...")
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "simulated_ride_data.csv"))
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. PREPARE DATA: AGGREGATE BY HOUR
# We need strictly 1-hour intervals.
print("📊 Aggregating data...")
df.set_index('timestamp', inplace=True)

# Main Target: Count of rides
hourly_demand = df.resample('H').size().to_frame(name='demand')

# Exogenous Features: Average Traffic, Weather Mode
# We need to aggregating categorical features.
# For simplicity, we'll take the mode (most frequent) weather per hour.
hourly_features = df.resample('H').agg({
    'weather': lambda x: x.mode()[0] if not x.empty else 'Sunny',
    'traffic_condition': lambda x: x.mode()[0] if not x.empty else 'Low'
})

# Merge
data = pd.concat([hourly_demand, hourly_features], axis=1).fillna(0)

# Feature Engineering on Aggregated Data
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# One-Hot Encode Weather & Traffic
data = pd.get_dummies(data, columns=['weather', 'traffic_condition'], drop_first=False)

# Ensure all expected columns exist (for consistent model input)
expected_cols = [
    'weather_Sunny', 'weather_Cloudy', 'weather_Rainy',
    'traffic_condition_Free Flow', 'traffic_condition_Low', 'traffic_condition_Medium',
    'traffic_condition_High', 'traffic_condition_Heavy Rain'
]
for col in expected_cols:
    if col not in data.columns:
        data[col] = 0

# Select Final Feature Set
# Target is 'demand', others are inputs. IMPORTANT: 'demand' must be first column for create_sequences logic
feature_cols = ['demand', 'hour', 'day_of_week', 'is_weekend'] + [c for c in data.columns if 'weather_' in c or 'traffic_' in c]
final_data = data[feature_cols].copy()

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(final_data)

# Create Sequences (Multivariate)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        # Input: All features for the window
        x = data[i:i+seq_length]
        # Output: Only the target (demand) for the NEXT step. Demand is at index 0.
        y = data[i+seq_length][0] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

print("Shape of data:", data_scaled.shape)
if len(data_scaled) < LOOKBACK_WINDOW + 2:
    print("⚠️ Not enough data for sequences. Duplicating data for training test...")
    data_scaled = np.concatenate([data_scaled] * 5, axis=0) # Fake it for demo if short

X, y = create_sequences(data_scaled, LOOKBACK_WINDOW)

# Convert to Tensor
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).view(-1, 1)

# 2. DEFINE LSTM MODEL (Multivariate)
class RideDemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(RideDemandLSTM, self).__init__()
        self.hidden_size = hidden_size
        # input_size = number of features (demand + weather + etc)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

input_dim = X_train.shape[2] # Number of features
print(f"🧠 Training LSTM with Input Dimension: {input_dim}")

model = RideDemandLSTM(input_dim, HIDDEN_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. TRAIN
print(f"🚀 Training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}')

# 4. SAVE MODEL & SCALER
print("Saving Multivariate Model...")
torch.save(model.state_dict(), "demand_model.pth")
joblib.dump(scaler, "scaler.pkl") # Save scaler to normalize inputs during inference
# Also save feature names to ensure alignment
joblib.dump(feature_cols, "model_features.pkl") 

print(f"✅ Multivariate Demand Model Saved! Inputs: {feature_cols}")