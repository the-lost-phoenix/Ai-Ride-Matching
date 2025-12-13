# 🚖 AI-Powered Vehicle Matching & Dynamic Pricing System

## 📌 Project Overview
An intelligent ride-hailing backend that predicts ETAs, forecasts demand using Deep Learning (LSTM), and optimizes pricing dynamically.

## 🚀 Key Features
* **ETA Prediction:** Gradient Boosting Regressor (LightGBM/XGBoost) to estimate travel time.
* **Demand Forecasting:** LSTM Neural Network (PyTorch) to predict hourly ride demand.
* **Dynamic Pricing:** Real-time surge pricing engine based on supply-demand ratios.
* **Vehicle Ranking:** Smart matching algorithm balancing User Preference (Cost vs. Time).

## 🛠️ Tech Stack
* **AI/ML:** PyTorch, Scikit-Learn, Pandas, NumPy
* **Backend:** FastAPI
* **Frontend:** Streamlit

## ⚙️ How to Run
1.  Clone the repo.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the API: `uvicorn main:app --reload`
4.  Run the UI: `streamlit run app.py`

## 📊 Model Performance
* **ETA Model MAE:** ~1.6 mins
* **Demand LSTM MAE:** ~2.1 rides/hr