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

    st.header("🚗 Predict Accident Probability")

    # Input widgets
    rider_age = st.number_input("Rider Age", min_value=12, max_value=100, value=26)
    rider_experience_years = st.number_input("Years of Riding Experience", min_value=0, max_value=80, value=8)
    speed = st.number_input("Speed (km/h)", min_value=0.0, max_value=120.0, value=40.0)
    helmet_used = st.selectbox("Helmet Used?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    alcohol_detected = st.selectbox("Alcohol Detected?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    recent_violations_count = st.number_input("Recent Violations Count", min_value=0, max_value=50, value=0)
    road_surface = st.selectbox("Road Surface", options=['Dry', 'Wet', 'Gravel', 'Mud'], index=2)
    light_condition = st.selectbox("Light Condition", options=['Daylight', 'Night', 'Dusk', 'Dawn'], index=2)
    brake_condition = st.slider("Brake Condition (0=worst, 1=best)", 0.0, 1.0, 0.9)
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
    st.markdown(f"""
    <div style='background-color:#f0f8ff; padding:10px; border-radius:10px;'>
    <h2 style='color:#007acc;'>🚨 Your Motorcycle Risk Factor:</h2>
    <h1 style='color:#d90429; font-weight:bold;'>{prediction[0]:.4f}</h1>
    <p style='font-style: italic;'>Stay safe on the road!</p>
    </div>
    """, unsafe_allow_html=True)

