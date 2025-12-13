import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- CONFIGURATION ---
LOOKBACK_WINDOW = 24  # The model looks at the last 24 hours to predict the next 1
HIDDEN_SIZE = 64      # Neurons in the LSTM layer
EPOCHS = 50           # How many times to practice
LEARNING_RATE = 0.001

print("⏳ Loading data for Demand Forecasting...")
df = pd.read_csv("simulated_ride_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. PREPARE DATA: AGGREGATE BY HOUR
# We need to count rides per hour to create a time-series
print("📊 Aggregating data by hour...")
df.set_index('timestamp', inplace=True)
hourly_demand = df.resample('H').size().to_frame(name='demand')

# Fill missing hours with 0 (e.g., if no rides at 3 AM)
hourly_demand = hourly_demand.fillna(0)

# 2. NORMALIZE DATA
# LSTMs work best when numbers are between 0 and 1
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(hourly_demand[['demand']].values)

# Save the scaler so we can reverse the math later (to get real ride counts)
joblib.dump(scaler, "demand_scaler.pkl")

# 3. CREATE SEQUENCES (The "Sliding Window")
# X = Past 24 hours, y = Next hour
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X_numpy, y_numpy = create_sequences(data_normalized, LOOKBACK_WINDOW)

# Convert to PyTorch Tensors
X_tensor = torch.FloatTensor(X_numpy)
y_tensor = torch.FloatTensor(y_numpy)

# Split into Train/Test (Keep last 100 hours for testing)
train_size = len(X_tensor) - 100
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

print(f"✅ Data Prepared. Training on {len(X_train)} sequences.")

# 4. DEFINE THE LSTM MODEL
class DemandLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(DemandLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM Layer
        out, _ = self.lstm(x)
        # We only care about the output of the LAST time step
        out = out[:, -1, :] 
        # Fully Connected Layer (Prediction)
        out = self.fc(out)
        return out

model = DemandLSTM(hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. TRAIN THE MODEL
print("🧠 Training LSTM (Deep Learning)...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    output = model(X_train)
    loss = criterion(output, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.5f}")

# 6. EVALUATE
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    # Convert back to real numbers (inverse transform)
    test_predictions_real = scaler.inverse_transform(test_predictions.numpy())
    y_test_real = scaler.inverse_transform(y_test.numpy())
    
    # Calculate Error
    mae = np.mean(np.abs(test_predictions_real - y_test_real))

print("-" * 30)
print(f"✅ LSTM Training Complete!")
print(f"📊 Mean Absolute Error (MAE): {mae:.2f} rides")
print(f"   (On average, we miss the demand count by {mae:.2f} rides)")
print("-" * 30)

# 7. SAVE THE MODEL
torch.save(model.state_dict(), "demand_lstm.pth")
print("💾 Model saved as 'demand_lstm.pth'")