# 🚖 AI Ride-Sharing System

A sophisticated ride-hailing application powered by **Machine Learning** models (LSTM + Gradient Boosting) for intelligent ETA prediction, demand forecasting, and dynamic surge pricing.

![AI Ride System](https://img.shields.io/badge/AI-Powered-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![React](https://img.shields.io/badge/React-19.0-61dafb) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)

## 🌟 Features

### **Frontend (React + Vite)**
- 🗺️ **Interactive Leaflet Maps** with draggable markers
- 📍 **Real-time Location Autocomplete** (Ola/Uber-style dropdown)
- 🎨 **Modern Ola-inspired UI** with Tailwind CSS
- 📱 **Bottom Sheet Interface** with smooth animations
- 🚗 **Animated Vehicle Cards** with surge pricing indicators
- ✅ **Booking Flow** with 3-stage confirmation animation
- 🔥 **Live Demand Heatmap** visualization

### **Backend (FastAPI + AI Models)**
- 🧠 **LSTM Neural Network** for demand forecasting
- ⚡ **Gradient Boosting Regressor** for ETA prediction
- 💰 **Dynamic Surge Pricing** based on real-time demand
- 🎯 **Smart Vehicle Ranking** (cheapest, fastest, balanced)
- 📊 **RESTful API** with automatic documentation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  (Leaflet Maps + Tailwind + Framer Motion)              │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST API
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                         │
│              (CORS + Pydantic Validation)                │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────┐         ┌────────▼────────┐
│ ETA Model    │         │  Demand Model   │
│ (GB Regressor)│         │  (LSTM PyTorch) │
│ eta_model.pkl│         │demand_lstm.pth  │
└──────────────┘         └─────────────────┘
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- Node.js 18+
- Git

### **1. Clone the Repository**
```bash
git clone <your-repo-url>
cd AI_Ride_System
```

### **2. Backend Setup**
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt

# Train models (if not already trained)
python train_eta_model.py
python train_demand_model.py

# Start backend server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend will run at: `http://127.0.0.1:8000`
API Docs: `http://127.0.0.1:8000/docs`

### **3. Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at: `http://localhost:5173`

## 📊 AI Models Explained

### **1. ETA Prediction Model (Gradient Boosting)**
**File:** `backend/eta_model.pkl`

**Training:** `train_eta_model.py`
- **Algorithm:** Gradient Boosting Regressor
- **Features:** `[trip_distance_km, traffic_multiplier, hour, day_of_week]`
- **Target:** Trip duration in minutes
- **Accuracy:** MAE ~2-3 minutes

**How it works:**
```python
Input: {
  distance: 5.2 km,
  traffic: 1.5x (rush hour),
  hour: 9 (9 AM),
  day: 2 (Wednesday)
}
Output: ETA = 18.5 minutes
```

### **2. Demand Forecasting Model (LSTM)**
**File:** `backend/demand_lstm.pth`

**Training:** `train_demand_model.py`
- **Algorithm:** LSTM Neural Network (PyTorch)
- **Input:** Last 24 hours of ride counts
- **Output:** Next hour ride demand
- **Architecture:** LSTM (64 hidden units) → Fully Connected
- **Accuracy:** MAE ~5-10 rides

**How it works:**
```python
Input: [45, 67, 32, 89, ...] # Last 24 hours
         ↓ LSTM learns patterns
Output: 78 rides expected next hour
         ↓ Surge calculation
Surge: 1.5x (because demand > 75)
```

### **3. Dynamic Pricing Formula**
```python
BASE_FARE = ₹50
PER_KM = ₹12
PER_MIN = ₹2

Price = (BASE_FARE + distance × 12 + eta × 2) × surge_multiplier

Example:
Price = (50 + 5.2×12 + 18.5×2) × 1.5
      = (50 + 62.4 + 37) × 1.5
      = ₹224
```

## 🎯 API Endpoints

### **POST /ride/quote**
Get ride quotes with AI predictions

**Request:**
```json
{
  "origin_lat": 12.9716,
  "origin_lon": 77.5946,
  "dest_lat": 12.9352,
  "dest_lon": 77.6245,
  "user_preference": "balanced"
}
```

**Response:**
```json
{
  "trip_distance_km": 5.2,
  "predicted_demand_next_hour": 78,
  "surge_applied": "1.5x",
  "options": [
    {
      "type": "Standard",
      "price": 224,
      "eta": 18.5,
      "surge": 1.5
    },
    ...
  ]
}
```

## 🛠️ Tech Stack

### **Frontend**
- **React 19** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling (via CDN)
- **Leaflet** - Interactive maps
- **Framer Motion** - Animations
- **Axios** - HTTP client
- **Lucide React** - Icons

### **Backend**
- **FastAPI** - Web framework
- **PyTorch** - Deep learning (LSTM)
- **Scikit-learn** - Machine learning (Gradient Boosting)
- **Pandas** - Data processing
- **Joblib** - Model serialization

## 📁 Project Structure

```
AI_Ride_System/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── pricing_engine.py          # AI model loader
│   ├── train_eta_model.py         # ETA model training
│   ├── train_demand_model.py      # Demand model training
│   ├── data_generator.py          # Synthetic data creation
│   ├── eta_model.pkl              # Trained ETA model
│   ├── demand_lstm.pth            # Trained LSTM model
│   ├── demand_scaler.pkl          # Data normalizer
│   └── simulated_ride_data.csv    # Training data
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Map.jsx            # Leaflet map
│   │   │   ├── BottomSheet.jsx    # Drawer interface
│   │   │   ├── VehicleCard.jsx    # Vehicle options
│   │   │   ├── BookingModal.jsx   # Booking confirmation
│   │   │   └── LocationAutocomplete.jsx  # Search dropdown
│   │   ├── App.jsx                # Main app
│   │   ├── App.css                # Styles
│   │   └── main.jsx               # Entry point
│   ├── index.html
│   └── package.json
│
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## 🎨 UI Features

### **Autocomplete Search**
- Type "indira" → Shows Indiranagar, Indira Nagar Metro, etc.
- Bangalore-specific filtering
- Keyboard navigation (↑↓ Enter Esc)
- 300ms debounced search

### **Map Interactions**
- Draggable pickup/drop pins
- Animated route visualization
- Nearby car markers with bounce animation
- Auto-zoom to fit route

### **Surge Pricing Visualization**
- Green (1.0x) - Normal demand
- Orange (1.2x) - Medium demand
- Red (1.5x+) - High demand

## 📝 Model Training

If you want to retrain models with your own data:

```bash
cd backend

# Generate new synthetic data
python data_generator.py

# Train ETA model
python train_eta_model.py

# Train demand model
python train_demand_model.py
```

## 🔍 How It Works (Complete Flow)

1. **User selects locations** via autocomplete
2. **Backend calculates distance** (Euclidean approximation)
3. **ETA Model predicts** trip duration based on distance, traffic, time
4. **LSTM forecasts demand** for next hour using 24-hour history
5. **Surge multiplier applied** based on demand thresholds
6. **Price calculated** using formula: Base + Distance + Time × Surge
7. **3 vehicle options** generated (Standard, Premium, Eco)
8. **Frontend displays** with animations

## 🌍 Location Restrictions

Currently configured for **Bangalore, India**:
- Latitude: 12.8° - 13.2°
- Longitude: 77.4° - 77.8°

To change city, modify coordinates in:
- `frontend/src/components/LocationAutocomplete.jsx` (line 52-56)
- `frontend/src/App.jsx` (line 26-28)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - feel free to use this for learning or commercial projects!

## 🙏 Acknowledgments

- OpenStreetMap for geocoding API
- Leaflet for map library
- Ola & Uber for UX inspiration

---

**Built with ❤️ using AI and modern web technologies**
