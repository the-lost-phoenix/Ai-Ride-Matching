import streamlit as st
import requests
import json

# --- CONFIGURATION ---
API_URL = "https://ai-ride-api.onrender.com/ride/quote"

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Ride Matcher", layout="wide")

st.title("🚖 AI-Powered Vehicle Matching System")
st.markdown("Predicts ETA, forecasts demand, and optimizes price using **LSTM & XGBoost**.")

# --- SIDEBAR (User Inputs) ---
with st.sidebar:
    st.header("📍 Book a Ride")
    
    # 1. Location Inputs (For demo, we use simple sliders/inputs or preset cities)
    # In a real app, this would be a map pin.
    st.subheader("Origin")
    origin_lat = st.number_input("Origin Latitude", value=12.9716, format="%.4f")
    origin_lon = st.number_input("Origin Longitude", value=77.5946, format="%.4f")
    
    st.subheader("Destination")
    # Default: A spot about 5km away
    dest_lat = st.number_input("Dest Latitude", value=12.9352, format="%.4f")
    dest_lon = st.number_input("Dest Longitude", value=77.6245, format="%.4f")
    
    st.subheader("Preferences")
    preference = st.selectbox("What matters most?", ["balanced", "cheapest", "fastest"])
    
    if st.button("🚀 Find My Ride"):
        # Prepare the payload for the API
        payload = {
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "dest_lat": dest_lat,
            "dest_lon": dest_lon,
            "user_preference": preference
        }
        
        # Call the API
        try:
            with st.spinner("Talking to AI Brain..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # SAVE DATA TO SESSION STATE (To show in main area)
                st.session_state['data'] = data
            else:
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Failed. Is the API running? Error: {e}")

# --- MAIN AREA (Results) ---
if 'data' in st.session_state:
    data = st.session_state['data']
    
    # 1. METRICS ROW
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 Trip Distance", f"{data['trip_distance_km']} km")
    col2.metric("🔮 Forecasted Demand", f"{data['predicted_demand_next_hour']} rides/hr")
    col3.metric("⚡ Surge Multiplier", data['surge_applied'])
    
    st.divider()
    
    # 2. VEHICLE OPTIONS
    st.subheader("🚗 Available Vehicles")
    
    # Create 3 columns for the cards
    cols = st.columns(3)
    
    for idx, option in enumerate(data['options']):
        with cols[idx]:
            # Card Styling
            st.info(f"**{option['type']}**")
            st.write(f"⏱️ ETA: **{option['eta']} mins**")
            st.write(f"💰 Price: **₹{option['price']}**")
            
            if st.button(f"Book {option['type']}", key=idx):
                st.success(f"🎉 Booking Confirmed for {option['type']}!")
                st.balloons()

else:
    st.info("👈 Use the sidebar to book a ride!")
    
    # SHOW A MAP (Visual Filler)
    # Just to make it look cool before they search
    import pandas as pd
    import numpy as np
    
    # Fake map data centered on Bangalore
    map_data = pd.DataFrame(
        np.random.randn(100, 2) / [50, 50] + [12.9716, 77.5946],
        columns=['lat', 'lon'])
    
    st.map(map_data)