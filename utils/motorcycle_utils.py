import streamlit as st
import pandas as pd
import pickle

def render_regression_tab():
    @st.cache_resource
    def load_model():
        with open("models/motorcycle-risk-predictor.pkl", "rb") as f:
            model = pickle.load(f)
        return model

    loaded_model = load_model()

    st.markdown("""
    <div style='padding:15px; border-radius:10px; margin-bottom:20px; background-color:#f9f9f9;'>
    <h2>Motorcycle Risk Predictor</h2>
    <p style='font-size:16px;'>
        This app estimates your motorcycle accident risk based on your riding habits, environment, 
        and safety factors. It leverages XGBoost machine learning model trained on extensive data 
        to help you make safer riding decisions. Input your ride details and get an instant, easy-to-understand risk score — helping you stay protected on every journey.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Input widgets
    rider_age = st.slider("Rider Age (year)", min_value=12, max_value=80, value=26, step=1)
    # Acceptable range for rider experience is 0 to 80 years based on

    rider_experience_years = st.number_input("Years of Riding Experience", min_value=1, max_value=rider_age, value=8)
    speed = st.slider("Speed (km/h)", min_value=0, max_value=120, value=40, step=1)
    helmet_used = st.selectbox("Helmet Used?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    alcohol_detected = st.selectbox("Alcohol Detected?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    recent_violations_count = st.number_input("Recent Violations Count", min_value=0, max_value=50, value=0)
    road_surface = st.selectbox("Road Surface", options=['Dry', 'Wet', 'Gravel', 'Mud'], index=2)
    light_condition = st.selectbox("Light Condition", options=['Daylight', 'Night', 'Dusk', 'Dawn'], index=2)
    brake_condition = st.slider("Brake Condition (0=worst, 1=best)",min_value=0.0,max_value=1.0,value=0.9,step=0.1)    
    weather_condition = st.selectbox("Weather Condition", options=['Clear', 'Rainy', 'Foggy', 'Windy', 'Snowy'], index=0)
    road_type = st.selectbox("Road Type", options=['Urban', 'Highway', 'Rural', 'Suburban'], index=1)

    # Prepare input dataframe
    input_data = pd.DataFrame({
        'rider_age': [rider_age],
        'rider_experience_years': [rider_experience_years],
        'speed': [speed],
        'helmet_used': [helmet_used],
        'alcohol_detected': [alcohol_detected],
        'recent_violations_count': [recent_violations_count],
        'road_surface': [road_surface],
        'light_condition': [light_condition],
        'brake_condition': [brake_condition],
        'weather_condition': [weather_condition],
        'road_type': [road_type]
    })

    prediction = loaded_model.predict(input_data)
    risk_percent = prediction[0] * 100

    # Define color based on risk percentage
    if risk_percent < 30:
        color = '#2ecc71'  # Green
    elif risk_percent < 70:
        color = '#f1c40f'  # Yellow
    else:
        color = '#e74c3c'  # Red

    st.markdown(f"""
    <div style='background-color:#f0f8ff; padding:10px; border-radius:10px;'>
    <h2 style='color:#007acc;'>Accident Risk Factor:</h2>
    <h1 style='color:{color}; font-weight:bold'>{risk_percent:.2f}%</h1>
    <p style='font-style: italic;'>Stay safe on the road!</p>
    </div>
    """, unsafe_allow_html=True)

